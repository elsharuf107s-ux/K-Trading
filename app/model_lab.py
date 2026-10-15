from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import settings
from .models import FeatureVector, Forecast
from .storage import Storage


FEATURE_NAMES = [
    'market_probability',
    'seconds_fraction',
    'spread',
    'momentum_10s',
    'momentum_30s',
    'momentum_60s',
    'momentum_180s',
    'reference_momentum_10s',
    'reference_momentum_30s',
    'reference_momentum_60s',
    'realized_vol_60s',
    'reference_vol_60s',
    'source_dispersion',
    'distance_to_target_bps',
    'book_imbalance',
    'market_momentum_10s',
    'market_momentum_30s',
    'wave_score',
    'official_reference',
    'log_yes_depth',
    'log_no_depth',
]


def _clip_probability(value: float) -> float:
    return max(1e-5, min(1 - 1e-5, float(value)))


def feature_vector(features: FeatureVector | dict[str, Any]) -> list[float]:
    f = features.model_dump() if isinstance(features, FeatureVector) else features
    official = str(f.get('reference_source') or '').startswith('cfbenchmarks')
    distance = f.get('distance_to_target_bps')
    return [
        float(f.get('market_probability') or 0.5),
        max(0.0, min(1.5, float(f.get('seconds_remaining') or 0.0) / 900.0)),
        float(f.get('spread') or 0.0),
        float(f.get('momentum_10s') or 0.0),
        float(f.get('momentum_30s') or 0.0),
        float(f.get('momentum_60s') or 0.0),
        float(f.get('momentum_180s') or 0.0),
        float(f.get('reference_momentum_10s') or 0.0),
        float(f.get('reference_momentum_30s') or 0.0),
        float(f.get('reference_momentum_60s') or 0.0),
        float(f.get('realized_vol_60s') or 0.0),
        float(f.get('reference_vol_60s') or 0.0),
        float(f.get('source_dispersion') or 0.0),
        float(distance if distance is not None else 0.0),
        float(f.get('book_imbalance') or 0.0),
        float(f.get('market_momentum_10s') or 0.0),
        float(f.get('market_momentum_30s') or 0.0),
        float(f.get('wave_score') or 0.0),
        1.0 if official else 0.0,
        math.log1p(max(0.0, float(f.get('yes_depth') or 0.0))),
        math.log1p(max(0.0, float(f.get('no_depth') or 0.0))),
    ]


@dataclass
class TrainingRow:
    created_at: str
    ticker: str
    outcome: int
    market_probability: float
    features: list[float]


class SigmoidCalibrator:
    """Small holdout calibrator that maps raw probabilities to calibrated probabilities."""

    def __init__(self):
        self.model: LogisticRegression | None = None

    def fit(self, probabilities: Iterable[float], outcomes: Iterable[int], sample_weight=None) -> 'SigmoidCalibrator':
        p = np.asarray([_clip_probability(x) for x in probabilities], dtype=float)
        y = np.asarray(list(outcomes), dtype=int)
        if len(p) < 6 or len(np.unique(y)) < 2:
            return self
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
        model.fit(logits, y, sample_weight=sample_weight)
        self.model = model
        return self

    def transform(self, probabilities: Iterable[float]) -> np.ndarray:
        p = np.asarray([_clip_probability(x) for x in probabilities], dtype=float)
        if self.model is None:
            return p
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


@dataclass
class ProbabilityModelBundle:
    name: str
    estimator: Any
    calibrator: SigmoidCalibrator
    feature_names: list[str]
    trained_at: str
    training_windows: int
    validation_metrics: dict[str, float]

    def predict_probability(self, features: FeatureVector | dict[str, Any]) -> float:
        x = np.asarray([feature_vector(features)], dtype=float)
        raw = float(self.estimator.predict_proba(x)[0, 1])
        calibrated = float(self.calibrator.transform([raw])[0])
        return max(0.01, min(0.99, calibrated))


class ModelRegistry:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or settings.model_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.candidate_dir = self.root / 'candidates'
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.champion_path = self.root / 'champion.pkl'
        self.champion_meta_path = self.root / 'champion.json'
        self.last_training_path = self.root / 'last_training.json'
        self._cached_champion: ProbabilityModelBundle | None = None
        self._cached_mtime: float | None = None

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')

    def record_training(self, report: dict[str, Any]) -> None:
        self._write_json(self.last_training_path, report)

    def champion_metadata(self) -> dict[str, Any] | None:
        if not self.champion_meta_path.exists():
            return None
        try:
            data = json.loads(self.champion_meta_path.read_text(encoding='utf-8'))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def load_champion(self) -> ProbabilityModelBundle | None:
        if not self.champion_path.exists():
            self._cached_champion = None
            self._cached_mtime = None
            return None
        mtime = self.champion_path.stat().st_mtime
        if self._cached_champion is not None and self._cached_mtime == mtime:
            return self._cached_champion
        try:
            with self.champion_path.open('rb') as fh:
                bundle = pickle.load(fh)
        except Exception:
            return None
        if not isinstance(bundle, ProbabilityModelBundle) or bundle.feature_names != FEATURE_NAMES:
            return None
        self._cached_champion = bundle
        self._cached_mtime = mtime
        return bundle

    def save_candidate(self, bundle: ProbabilityModelBundle, report: dict[str, Any]) -> Path:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        safe_name = bundle.name.replace('/', '_').replace(' ', '_')
        path = self.candidate_dir / f'{stamp}-{safe_name}.pkl'
        with path.open('wb') as fh:
            pickle.dump(bundle, fh)
        meta = dict(report)
        meta['artifact'] = str(path)
        self._write_json(path.with_suffix('.json'), meta)
        return path

    def promote(self, bundle: ProbabilityModelBundle, report: dict[str, Any]) -> None:
        tmp = self.root / 'champion.tmp.pkl'
        with tmp.open('wb') as fh:
            pickle.dump(bundle, fh)
        tmp.replace(self.champion_path)
        metadata = dict(report)
        metadata.update({
            'name': bundle.name,
            'trained_at': bundle.trained_at,
            'training_windows': bundle.training_windows,
            'feature_names': FEATURE_NAMES,
        })
        self._write_json(self.champion_meta_path, metadata)
        self._cached_champion = bundle
        self._cached_mtime = self.champion_path.stat().st_mtime

    def status(self) -> dict[str, Any]:
        champion = self.champion_metadata()
        last = None
        if self.last_training_path.exists():
            try:
                last = json.loads(self.last_training_path.read_text(encoding='utf-8'))
            except Exception:
                last = None
        return {
            'enabled': settings.enable_learned_model,
            'champion': champion,
            'last_training': last,
            'model_dir': str(self.root),
        }


class AdaptiveForecaster:
    """Uses the promoted learned champion when available; otherwise falls back safely."""

    def __init__(self, fallback, registry: ModelRegistry | None = None):
        self.fallback = fallback
        self.registry = registry or ModelRegistry()

    def predict(self, f: FeatureVector) -> Forecast:
        if not settings.enable_learned_model:
            return self.fallback.predict(f)
        champion = self.registry.load_champion()
        if champion is None:
            return self.fallback.predict(f)
        try:
            probability = champion.predict_probability(f)
        except Exception:
            return self.fallback.predict(f)
        confidence = max(0.30, min(0.90, 0.45 + abs(probability - 0.5) * 0.7))
        return Forecast(
            yes_probability=probability,
            confidence=confidence,
            rationale=[
                f'Promoted learned model: {champion.name}',
                f'Validation Brier {champion.validation_metrics.get("brier", float("nan")):.4f}',
                f'Wave {f.wave_state} ({f.wave_score:+.2f})',
                f'Reference source: {f.reference_source}',
            ],
            model_name=f'learned:{champion.name}',
        )


class ModelLab:
    def __init__(self, storage: Storage, registry: ModelRegistry | None = None):
        self.storage = storage
        self.registry = registry or ModelRegistry()

    def _rows(self) -> list[TrainingRow]:
        cur = self.storage.db.execute(
            '''SELECT o.created_at,o.ticker,o.usable_json,r.result
               FROM observations o JOIN market_results r ON r.ticker=o.ticker
               ORDER BY o.created_at ASC,o.id ASC'''
        )
        rows: list[TrainingRow] = []
        for created_at, ticker, payload_json, result in cur.fetchall():
            try:
                payload = json.loads(payload_json)
                f = payload['features']
                vector = feature_vector(f)
                market_probability = float(f.get('market_probability') or 0.5)
            except Exception:
                continue
            rows.append(TrainingRow(
                created_at=str(created_at),
                ticker=str(ticker),
                outcome=1 if str(result).lower() == 'yes' else 0,
                market_probability=market_probability,
                features=vector,
            ))
        return rows

    @staticmethod
    def _ordered_windows(rows: list[TrainingRow]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for row in rows:
            if row.ticker not in seen:
                seen.add(row.ticker)
                ordered.append(row.ticker)
        return ordered

    @staticmethod
    def _select(rows: list[TrainingRow], tickers: set[str]):
        chosen = [r for r in rows if r.ticker in tickers]
        x = np.asarray([r.features for r in chosen], dtype=float)
        y = np.asarray([r.outcome for r in chosen], dtype=int)
        market = np.asarray([r.market_probability for r in chosen], dtype=float)
        counts: dict[str, int] = {}
        for row in chosen:
            counts[row.ticker] = counts.get(row.ticker, 0) + 1
        weights = np.asarray([1.0 / counts[r.ticker] for r in chosen], dtype=float)
        return chosen, x, y, market, weights

    @staticmethod
    def _factory(name: str):
        if name == 'logistic':
            return Pipeline([
                ('scale', StandardScaler()),
                ('clf', LogisticRegression(C=0.7, max_iter=2000, class_weight='balanced', random_state=42)),
            ])
        if name == 'hist_gradient_boosting':
            return HistGradientBoostingClassifier(
                loss='log_loss',
                learning_rate=0.05,
                max_iter=180,
                max_leaf_nodes=15,
                min_samples_leaf=8,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=42,
            )
        raise ValueError(f'Unknown candidate model: {name}')

    @staticmethod
    def _fit(model, x, y, weights):
        if isinstance(model, Pipeline):
            model.fit(x, y, clf__sample_weight=weights)
        else:
            model.fit(x, y, sample_weight=weights)
        return model

    @staticmethod
    def _metrics(y, probabilities, market_probabilities, weights) -> dict[str, float]:
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-5, 1 - 1e-5)
        m = np.clip(np.asarray(market_probabilities, dtype=float), 1e-5, 1 - 1e-5)
        return {
            'brier': float(brier_score_loss(y, p, sample_weight=weights)),
            'log_loss': float(log_loss(y, p, sample_weight=weights, labels=[0, 1])),
            'market_brier': float(brier_score_loss(y, m, sample_weight=weights)),
            'market_log_loss': float(log_loss(y, m, sample_weight=weights, labels=[0, 1])),
        }

    def _walk_forward(self, rows: list[TrainingRow], windows: list[str], candidate: str) -> dict[str, Any] | None:
        n = len(windows)
        min_train = max(8, n // 2)
        remaining = n - min_train
        if remaining < 2:
            return None
        folds = min(4, max(1, remaining // 2))
        chunk = max(1, remaining // folds)
        predictions: list[float] = []
        outcomes: list[int] = []
        markets: list[float] = []
        weights_all: list[float] = []
        fold_reports: list[dict[str, Any]] = []

        for index in range(folds):
            train_end = min_train + index * chunk
            test_end = n if index == folds - 1 else min(n, train_end + chunk)
            if train_end >= test_end:
                continue
            train_tickers = set(windows[:train_end])
            test_tickers = set(windows[train_end:test_end])
            _, x_train, y_train, _, w_train = self._select(rows, train_tickers)
            _, x_test, y_test, market_test, w_test = self._select(rows, test_tickers)
            if len(x_train) == 0 or len(x_test) == 0 or len(np.unique(y_train)) < 2:
                continue
            model = self._fit(self._factory(candidate), x_train, y_train, w_train)
            p = model.predict_proba(x_test)[:, 1]
            fold_metrics = self._metrics(y_test, p, market_test, w_test)
            fold_reports.append({
                'train_windows': len(train_tickers),
                'test_windows': len(test_tickers),
                **fold_metrics,
            })
            predictions.extend(float(x) for x in p)
            outcomes.extend(int(x) for x in y_test)
            markets.extend(float(x) for x in market_test)
            weights_all.extend(float(x) for x in w_test)

        if not predictions:
            return None
        aggregate = self._metrics(
            np.asarray(outcomes),
            np.asarray(predictions),
            np.asarray(markets),
            np.asarray(weights_all),
        )
        aggregate['brier_gain_vs_market'] = aggregate['market_brier'] - aggregate['brier']
        aggregate['relative_brier_improvement'] = (
            aggregate['brier_gain_vs_market'] / aggregate['market_brier']
            if aggregate['market_brier'] > 0 else 0.0
        )
        return {
            'name': candidate,
            'metrics': aggregate,
            'folds': fold_reports,
            'oos_snapshots': len(predictions),
        }

    def _fit_final_bundle(self, rows: list[TrainingRow], windows: list[str], candidate: str, metrics: dict[str, float]) -> ProbabilityModelBundle | None:
        calibration_windows = max(2, min(8, len(windows) // 5))
        if len(windows) - calibration_windows < 8:
            return None
        train_tickers = set(windows[:-calibration_windows])
        calibration_tickers = set(windows[-calibration_windows:])
        _, x_train, y_train, _, w_train = self._select(rows, train_tickers)
        _, x_cal, y_cal, _, w_cal = self._select(rows, calibration_tickers)
        if len(np.unique(y_train)) < 2 or len(x_train) == 0:
            return None
        model = self._fit(self._factory(candidate), x_train, y_train, w_train)
        calibrator = SigmoidCalibrator()
        if len(x_cal):
            raw_cal = model.predict_proba(x_cal)[:, 1]
            calibrator.fit(raw_cal, y_cal, sample_weight=w_cal)
        return ProbabilityModelBundle(
            name=candidate,
            estimator=model,
            calibrator=calibrator,
            feature_names=list(FEATURE_NAMES),
            trained_at=datetime.now(timezone.utc).isoformat(),
            training_windows=len(train_tickers),
            validation_metrics=dict(metrics),
        )

    def train(self) -> dict[str, Any]:
        rows = self._rows()
        windows = self._ordered_windows(rows)
        report: dict[str, Any] = {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'status': 'insufficient_data',
            'windows': len(windows),
            'snapshots': len(rows),
            'minimum_windows': settings.model_min_windows,
            'feature_names': FEATURE_NAMES,
            'candidates': [],
            'promoted': False,
        }
        if len(windows) < settings.model_min_windows:
            report['reason'] = f'Need at least {settings.model_min_windows} completed markets before training'
            self.registry.record_training(report)
            return report

        candidates = []
        for name in ('logistic', 'hist_gradient_boosting'):
            result = self._walk_forward(rows, windows, name)
            if result:
                candidates.append(result)
        report['candidates'] = candidates
        if not candidates:
            report['reason'] = 'Not enough class diversity for leak-resistant walk-forward evaluation'
            self.registry.record_training(report)
            return report

        candidates.sort(key=lambda item: (item['metrics']['brier'], item['metrics']['log_loss']))
        best = candidates[0]
        report['selected'] = best
        bundle = self._fit_final_bundle(rows, windows, best['name'], best['metrics'])
        if bundle is None:
            report['reason'] = 'Could not fit a final model with a disjoint calibration holdout'
            self.registry.record_training(report)
            return report

        candidate_path = self.registry.save_candidate(bundle, report)
        report['candidate_artifact'] = str(candidate_path)
        gain = float(best['metrics']['brier_gain_vs_market'])
        existing = self.registry.champion_metadata()
        threshold = settings.model_min_brier_gain
        if existing and isinstance(existing.get('validation_metrics'), dict):
            old_brier = float(existing['validation_metrics'].get('brier', 1.0))
            beats_champion = best['metrics']['brier'] + threshold / 2 < old_brier
        else:
            beats_champion = True
        beats_market = gain >= threshold

        if beats_market and beats_champion:
            promotion_report = dict(report)
            promotion_report['status'] = 'promoted'
            promotion_report['promoted'] = True
            promotion_report['validation_metrics'] = best['metrics']
            self.registry.promote(bundle, promotion_report)
            report = promotion_report
        else:
            report['status'] = 'candidate_only'
            report['reason'] = (
                f'Candidate Brier gain {gain:.5f} did not clear promotion rules '
                f'(market gain >= {threshold:.5f} and beat current champion)'
            )

        self.registry.record_training(report)
        return report
