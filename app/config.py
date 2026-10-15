from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_host: str = '0.0.0.0'
    app_port: int = 8000
    engine_interval_seconds: float = 2.0
    database_path: str = 'data/k_trading.sqlite3'

    kalshi_base_url: str = 'https://external-api.kalshi.com/trade-api/v2'
    kalshi_ws_url: str = 'wss://external-api-ws.kalshi.com/trade-api/ws/v2'
    kalshi_series_ticker: str = 'KXBTC15M'
    market_ticker_override: str = ''

    kalshi_key_id: str = ''
    kalshi_private_key_path: str = ''
    kalshi_private_key_pem: str = ''
    enable_kalshi_ws: bool = True
    cfbenchmarks_index_id: str = 'BRTI'
    benchmark_stale_seconds: float = 5.0

    paper_starting_cash: float = 1000.0
    auto_paper_trade: bool = True
    live_trading_enabled: bool = False

    min_usable_edge: float = 0.04
    exit_edge: float = 0.01
    uncertainty_buffer: float = 0.02
    estimated_cost_buffer: float = 0.01
    max_spread: float = 0.12
    max_position_usd: float = 15.0
    max_bankroll_fraction: float = 0.015
    min_seconds_to_expiry: int = 45
    official_reference_guard_seconds: int = 75
    min_wave_strength: float = 0.08
    max_trade_loss_fraction: float = 0.25
    trailing_profit_arm_fraction: float = 0.10
    trailing_giveback_fraction: float = 0.35
    daily_loss_limit_usd: float = 50.0
    max_drawdown_fraction: float = 0.10

    coinbase_spot_url: str = 'https://api.coinbase.com/v2/prices/BTC-USD/spot'
    kraken_ticker_url: str = 'https://api.kraken.com/0/public/Ticker?pair=XBTUSD'

    def ensure_dirs(self) -> None:
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)

    @property
    def has_kalshi_credentials(self) -> bool:
        return bool(self.kalshi_key_id and (self.kalshi_private_key_path or self.kalshi_private_key_pem))


settings = Settings()
