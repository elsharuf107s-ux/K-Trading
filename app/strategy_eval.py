from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from .config import settings
from .execution import ExecutionSimulator
from .models import FeatureVector, Forecast, MarketSnapshot
from .storage import Storage


def _safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def _bucket_edge(edge: float) -> str:
    if edge < 0.06:
        return '4-6%'
    if edge < 0.10:
        return '6-10%'
    if edge < 0.15:
        return '10-15%'
    return '15%+'


def _bucket_time(seconds: float) -> str:
    if seconds <= 180:
        return '0-3m'
    if seconds <= 450:
        return '3-7.5m'
    return '7.5-15m'


def _bucket_wave(score: float) -> str:
    strength = abs(score)
    if strength < 0.20:
        return 'weak'
    if strength < 0.50:
        return 'moderate'
    return 'strong'


def _bucket_reference(source: str) -> str:
    return 'official' if str(source).startswith('cfbenchmarks') else 'proxy'


class StrategyEvaluator:
    """Chronological, one-contract replay of the recorded decision frames.

    The final market result is never used to choose an entry or exit. It is
    only read after the frame replay finishes so any still-open contract can be
    settled and scored.
    """

    def __init__(self, storage: Storage, report_path: str | Path | None = None):
        self.storage = storage
        self.execution = ExecutionSimulator()
        self.report_path = Path(report_path or settings.strategy_eval_report_path)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

    def last_report(self) -> dict[str, Any] | None:
        if not self.report_path.exists():
            return None
        try:
            data = json.loads(self.report_path.read_text(encoding='utf-8'))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _completed_windows(self, limit: int = 100_000) -> list[dict[str, Any]]:
        rows = self.storage.db.execute(
            '''SELECT o.ticker,o.created_at,o.usable_json,r.result,r.settled_at
               FROM observations o JOIN market_results r ON r.ticker=o.ticker
               ORDER BY o.created_at ASC,o.id ASC LIMIT ?''',
            (int(limit),),
        ).fetchall()
        grouped: dict[str, dict[str, Any]] = {}
        for ticker, created_at, payload, result, settled_at in rows:
            item = grouped.setdefault(str(ticker), {
                'ticker': str(ticker),
                'result': str(result).lower(),
                'settled_at': settled_at,
                'frames': [],
            })
            try:
                parsed = json.loads(payload)
            except Exception:
                continue
            item['frames'].append({'created_at': str(created_at), **parsed})
        return [item for item in grouped.values() if item['frames'] and item['result'] in ('yes', 'no')]

    @staticmethod
    def _frame(frame: dict[str, Any]) -> tuple[MarketSnapshot, FeatureVector, Forecast] | None:
        try:
            return (
                MarketSnapshot.model_validate(frame['market']),
                FeatureVector.model_validate(frame['features']),
                Forecast.model_validate(frame['forecast']),
            )
        except Exception:
            return None

    def _entry_signal(self, market: MarketSnapshot, f: FeatureVector, forecast: Forecast) -> dict[str, Any] | None:
        if f.seconds_remaining < settings.min_seconds_to_expiry:
            return None
        if f.seconds_remaining <= settings.official_reference_guard_seconds and not f.reference_source.startswith('cfbenchmarks'):
            return None
        if f.spread > settings.max_spread or f.source_dispersion > settings.max_source_dispersion:
            return None

        friction = settings.estimated_cost_buffer + settings.uncertainty_buffer
        yes_edge = forecast.yes_probability - f.yes_ask - friction
        no_edge = (1.0 - forecast.yes_probability) - f.no_ask - friction
        side = 'yes' if yes_edge >= no_edge else 'no'
        raw_edge = yes_edge if side == 'yes' else no_edge
        if raw_edge < settings.min_usable_edge:
            return None
        if side == 'yes' and f.wave_score < settings.min_wave_strength:
            return None
        if side == 'no' and f.wave_score > -settings.min_wave_strength:
            return None
        if settings.require_orderbook_for_entry:
            executable_depth = f.no_depth if side == 'yes' else f.yes_depth
            if executable_depth < settings.min_execution_side_depth:
                return None

        fill = self.execution.simulate(market, side, 'buy', 1, wave_score=f.wave_score)
        if fill.filled_quantity < 1:
            return None
        fair = forecast.yes_probability if side == 'yes' else 1.0 - forecast.yes_probability
        fee_per_contract = fill.fee
        executable_edge = fair - fill.average_price - fee_per_contract - settings.uncertainty_buffer
        if executable_edge < settings.min_usable_edge:
            return None
        if fill.slippage > settings.sim_max_execution_slippage:
            return None

        return {
            'side': side,
            'fill': fill,
            'fair': fair,
            'raw_edge': raw_edge,
            'executable_edge': executable_edge,
        }

    def _should_exit(
        self,
        f: FeatureVector,
        forecast: Forecast,
        side: str,
        entry_price: float,
        peak_bid: float,
    ) -> tuple[bool, str, float]:
        bid = f.yes_bid if side == 'yes' else f.no_bid
        fair = forecast.yes_probability if side == 'yes' else 1.0 - forecast.yes_probability
        remaining_edge = fair - bid - settings.estimated_cost_buffer
        trade_return = bid / max(entry_price, 1e-9) - 1.0
        if trade_return <= -settings.max_trade_loss_fraction:
            return True, 'stop_loss', bid
        peak_return = peak_bid / max(entry_price, 1e-9) - 1.0
        if peak_return >= settings.trailing_profit_arm_fraction and peak_bid > entry_price:
            giveback = (peak_bid - bid) / max(peak_bid - entry_price, 1e-9)
            if giveback >= settings.trailing_giveback_fraction:
                return True, 'trailing_profit', bid
        wave_reversed = (
            (side == 'yes' and f.wave_score < -settings.min_wave_strength)
            or (side == 'no' and f.wave_score > settings.min_wave_strength)
        )
        if wave_reversed:
            return True, 'wave_reversal', bid
        if remaining_edge <= settings.exit_edge:
            return True, 'edge_decay', bid
        if f.seconds_remaining <= settings.official_reference_guard_seconds and not f.reference_source.startswith('cfbenchmarks'):
            return True, 'reference_guard', bid
        if f.seconds_remaining < settings.min_seconds_to_expiry:
            return True, 'time_exit', bid
        return False, 'hold', bid

    def _replay_window(self, window: dict[str, Any]) -> dict[str, Any] | None:
        position: dict[str, Any] | None = None
        last_frame: tuple[MarketSnapshot, FeatureVector, Forecast] | None = None

        for raw_frame in window['frames']:
            parsed = self._frame(raw_frame)
            if parsed is None:
                continue
            market, f, forecast = parsed
            last_frame = parsed

            if position is None:
                signal = self._entry_signal(market, f, forecast)
                if signal is None:
                    continue
                fill = signal['fill']
                position = {
                    'side': signal['side'],
                    'entry_price': fill.average_price,
                    'entry_fee': fill.fee,
                    'entry_seconds': f.seconds_remaining,
                    'entry_edge': signal['executable_edge'],
                    'gross_signal_edge': signal['raw_edge'],
                    'entry_wave_score': f.wave_score,
                    'entry_reference_source': f.reference_source,
                    'entry_model_probability': forecast.yes_probability,
                    'entry_market_probability': f.market_probability,
                    'entry_created_at': raw_frame.get('created_at'),
                    'entry_depth_slippage': fill.depth_slippage,
                    'entry_latency_slippage': fill.latency_slippage,
                    'peak_bid': f.yes_bid if signal['side'] == 'yes' else f.no_bid,
                }
                continue

            side = position['side']
            bid = f.yes_bid if side == 'yes' else f.no_bid
            position['peak_bid'] = max(position['peak_bid'], bid)
            should_exit, reason, _ = self._should_exit(
                f, forecast, side, position['entry_price'], position['peak_bid'],
            )
            if not should_exit:
                continue
            fill = self.execution.simulate(market, side, 'sell', 1, wave_score=f.wave_score)
            if fill.filled_quantity < 1:
                continue
            pnl = fill.average_price - position['entry_price'] - position['entry_fee'] - fill.fee
            return self._trade_result(window, position, pnl, fill.average_price, fill.fee, reason, f.seconds_remaining)

        if position is None:
            return None

        # Settlement is known only after all decision frames have been replayed.
        settle_price = 1.0 if window['result'] == position['side'] else 0.0
        pnl = settle_price - position['entry_price'] - position['entry_fee']
        exit_seconds = last_frame[1].seconds_remaining if last_frame else 0.0
        return self._trade_result(window, position, pnl, settle_price, 0.0, 'settlement', exit_seconds)

    @staticmethod
    def _trade_result(
        window: dict[str, Any],
        position: dict[str, Any],
        pnl: float,
        exit_price: float,
        exit_fee: float,
        exit_reason: str,
        exit_seconds: float,
    ) -> dict[str, Any]:
        side = position['side']
        direction_correct = window['result'] == side
        return {
            'ticker': window['ticker'],
            'result': window['result'],
            'side': side,
            'direction_correct': direction_correct,
            'entry_created_at': position['entry_created_at'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_fee': position['entry_fee'],
            'exit_fee': exit_fee,
            'total_fee': position['entry_fee'] + exit_fee,
            'pnl': pnl,
            'entry_seconds': position['entry_seconds'],
            'exit_seconds': exit_seconds,
            'hold_seconds': max(0.0, position['entry_seconds'] - exit_seconds),
            'entry_edge': position['entry_edge'],
            'gross_signal_edge': position['gross_signal_edge'],
            'execution_drag_at_entry': position['gross_signal_edge'] - position['entry_edge'],
            'entry_depth_slippage': position['entry_depth_slippage'],
            'entry_latency_slippage': position['entry_latency_slippage'],
            'entry_wave_score': position['entry_wave_score'],
            'entry_reference_source': position['entry_reference_source'],
            'entry_model_probability': position['entry_model_probability'],
            'entry_market_probability': position['entry_market_probability'],
            'exit_reason': exit_reason,
        }

    @staticmethod
    def _bucket_summary(trades: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trade in trades:
            groups[str(key_fn(trade))].append(trade)
        result: list[dict[str, Any]] = []
        for name, rows in groups.items():
            pnl = [float(r['pnl']) for r in rows]
            result.append({
                'bucket': name,
                'trades': len(rows),
                'net_pnl': sum(pnl),
                'average_pnl': sum(pnl) / len(pnl),
                'win_rate': sum(1 for x in pnl if x > 0) / len(pnl),
                'direction_accuracy': sum(1 for r in rows if r['direction_correct']) / len(rows),
                'average_entry_edge': sum(float(r['entry_edge']) for r in rows) / len(rows),
            })
        return sorted(result, key=lambda x: x['bucket'])

    @staticmethod
    def _equity_metrics(trades: list[dict[str, Any]], starting_equity: float = 100.0) -> dict[str, Any]:
        equity = starting_equity
        peak = equity
        max_drawdown = 0.0
        max_drawdown_fraction = 0.0
        current_losses = 0
        max_consecutive_losses = 0
        curve = []
        for trade in trades:
            pnl = float(trade['pnl'])
            equity += pnl
            peak = max(peak, equity)
            drawdown = peak - equity
            max_drawdown = max(max_drawdown, drawdown)
            max_drawdown_fraction = max(max_drawdown_fraction, drawdown / max(peak, 1e-9))
            if pnl < 0:
                current_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_losses = 0
            curve.append({'ticker': trade['ticker'], 'equity': equity, 'pnl': pnl})
        return {
            'starting_equity': starting_equity,
            'ending_equity': equity,
            'max_drawdown': max_drawdown,
            'max_drawdown_fraction': max_drawdown_fraction,
            'max_consecutive_losses': max_consecutive_losses,
            'equity_curve': curve,
        }

    def run(self) -> dict[str, Any]:
        windows = self._completed_windows()
        trades: list[dict[str, Any]] = []
        skipped = 0
        for window in windows:
            trade = self._replay_window(window)
            if trade is None:
                skipped += 1
                continue
            trades.append(trade)

        pnl = [float(t['pnl']) for t in trades]
        wins = [x for x in pnl if x > 0]
        losses = [x for x in pnl if x < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        fees = sum(float(t['total_fee']) for t in trades)
        equity = self._equity_metrics(trades)

        metrics = {
            'completed_windows': len(windows),
            'traded_windows': len(trades),
            'skipped_windows': skipped,
            'trade_rate': _safe_div(len(trades), len(windows)),
            'net_pnl_per_contract_strategy': sum(pnl),
            'average_pnl': _safe_div(sum(pnl), len(pnl)),
            'win_rate': _safe_div(len(wins), len(trades)),
            'direction_accuracy': _safe_div(sum(1 for t in trades if t['direction_correct']), len(trades)),
            'profit_factor': _safe_div(gross_profit, gross_loss) if gross_loss else ('inf' if gross_profit > 0 else None),
            'total_fees': fees,
            'average_fee': _safe_div(fees, len(trades)),
            'average_entry_edge': _safe_div(sum(float(t['entry_edge']) for t in trades), len(trades)),
            'average_execution_drag_at_entry': _safe_div(sum(float(t['execution_drag_at_entry']) for t in trades), len(trades)),
            'average_hold_seconds': _safe_div(sum(float(t['hold_seconds']) for t in trades), len(trades)),
            **{k: v for k, v in equity.items() if k != 'equity_curve'},
        }

        report = {
            'status': 'ok',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'methodology': {
                'unit_size': 'one contract per completed window',
                'chronology': 'frames replayed in recorded timestamp order',
                'settlement_usage': 'result used only after decision-frame replay',
                'execution': 'recorded visible book + current V0.5 execution simulator settings',
                'warning': 'Research replay is not live-trading evidence and does not model portfolio-level simultaneous exposure.',
            },
            'metrics': metrics,
            'buckets': {
                'entry_edge': self._bucket_summary(trades, lambda t: _bucket_edge(float(t['entry_edge']))),
                'entry_time': self._bucket_summary(trades, lambda t: _bucket_time(float(t['entry_seconds']))),
                'wave_strength': self._bucket_summary(trades, lambda t: _bucket_wave(float(t['entry_wave_score']))),
                'reference': self._bucket_summary(trades, lambda t: _bucket_reference(str(t['entry_reference_source']))),
                'side': self._bucket_summary(trades, lambda t: t['side']),
                'exit_reason': self._bucket_summary(trades, lambda t: t['exit_reason']),
            },
            'trades': trades[-settings.strategy_eval_max_report_trades:],
            'equity_curve': equity['equity_curve'][-settings.strategy_eval_max_report_trades:],
        }
        self.report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
        return report


class ResearchGate:
    """Development go/no-go gate. Passing is not a claim of safety or profit."""

    def __init__(self, storage: Storage, evaluator: StrategyEvaluator):
        self.storage = storage
        self.evaluator = evaluator

    def evaluate(
        self,
        analytics: dict[str, Any],
        empirical_summary: dict[str, Any],
        strategy_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        report = strategy_report or self.evaluator.last_report()
        metrics = (report or {}).get('metrics') or {}
        execution = analytics.get('execution') or {}
        brier_gain = analytics.get('brier_improvement_vs_market')
        completed = int(analytics.get('completed_windows') or 0)
        empirical_samples = int(empirical_summary.get('total_samples') or 0)
        # 0 is a perfect score for these two once trades exist; without
        # replay trades (or metrics) there is no evidence, so they fail.
        traded = int(metrics.get('traded_windows') or 0)
        drawdown = metrics.get('max_drawdown_fraction') if traded else None
        loss_streak = metrics.get('max_consecutive_losses') if traded else None

        gates = [
            {
                'name': 'completed_windows',
                'pass': completed >= settings.soak_min_completed_windows,
                'actual': completed,
                'required': settings.soak_min_completed_windows,
            },
            {
                'name': 'replay_trades',
                'pass': traded >= settings.soak_min_replay_trades,
                'actual': traded,
                'required': settings.soak_min_replay_trades,
            },
            {
                'name': 'brier_improvement',
                'pass': brier_gain is not None and float(brier_gain) >= settings.soak_min_brier_gain,
                'actual': brier_gain,
                'required': settings.soak_min_brier_gain,
            },
            {
                'name': 'replay_net_pnl',
                'pass': (
                    not settings.soak_require_positive_replay_pnl
                    or float(metrics.get('net_pnl_per_contract_strategy') or 0.0) > 0.0
                ),
                'actual': metrics.get('net_pnl_per_contract_strategy'),
                'required': '> 0' if settings.soak_require_positive_replay_pnl else 'not required',
            },
            {
                'name': 'replay_drawdown',
                'pass': drawdown is not None and float(drawdown) <= settings.soak_max_replay_drawdown_fraction,
                'actual': drawdown,
                'required': f'<= {settings.soak_max_replay_drawdown_fraction}',
            },
            {
                'name': 'loss_streak',
                'pass': loss_streak is not None and int(loss_streak) <= settings.soak_max_consecutive_losses,
                'actual': loss_streak,
                'required': f'<= {settings.soak_max_consecutive_losses}',
            },
            {
                'name': 'execution_fill_rate',
                'pass': float(execution.get('fill_rate') or 0.0) >= settings.soak_min_execution_fill_rate,
                'actual': execution.get('fill_rate'),
                'required': settings.soak_min_execution_fill_rate,
            },
            {
                'name': 'empirical_execution_samples',
                'pass': empirical_samples >= settings.soak_min_empirical_samples,
                'actual': empirical_samples,
                'required': settings.soak_min_empirical_samples,
            },
        ]
        passed = all(bool(g['pass']) for g in gates)
        return {
            'status': 'GO_FOR_MORE_PAPER_RESEARCH' if passed else 'NOT_READY',
            'passed': passed,
            'gates': gates,
            'note': 'This is a development research gate only. Passing does not authorize or prove safe/profitable live trading.',
        }
