from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from .models import FeatureVector, Forecast, MarketSnapshot, TradeRecord


class Storage:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            ticker TEXT NOT NULL,
            btc_price REAL NOT NULL,
            market_probability REAL NOT NULL,
            yes_bid REAL, yes_ask REAL, no_bid REAL, no_ask REAL,
            seconds_remaining REAL,
            model_probability REAL,
            confidence REAL,
            usable_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            pnl REAL NOT NULL,
            reason TEXT NOT NULL
        );
        ''')
        self.db.commit()

    def save_observation(self, market: MarketSnapshot, features: FeatureVector, forecast: Forecast) -> None:
        payload = {'market': market.model_dump(mode='json'), 'features': features.model_dump(mode='json'), 'forecast': forecast.model_dump(mode='json')}
        self.db.execute(
            'INSERT INTO observations(created_at,ticker,btc_price,market_probability,yes_bid,yes_ask,no_bid,no_ask,seconds_remaining,model_probability,confidence,usable_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (market.captured_at.isoformat(), market.ticker, features.btc_price, features.market_probability, features.yes_bid, features.yes_ask, features.no_bid, features.no_ask, features.seconds_remaining, forecast.yes_probability, forecast.confidence, json.dumps(payload))
        )
        self.db.commit()

    def save_trade(self, trade: TradeRecord) -> None:
        self.db.execute(
            'INSERT INTO trades(created_at,ticker,side,action,quantity,price,pnl,reason) VALUES(?,?,?,?,?,?,?,?)',
            (trade.created_at.isoformat(), trade.ticker, trade.side, trade.action, trade.quantity, trade.price, trade.pnl, trade.reason)
        )
        self.db.commit()

    def recent_trades(self, limit: int = 50) -> list[dict]:
        cur = self.db.execute('SELECT created_at,ticker,side,action,quantity,price,pnl,reason FROM trades ORDER BY id DESC LIMIT ?', (limit,))
        keys = ['created_at','ticker','side','action','quantity','price','pnl','reason']
        return [dict(zip(keys, row)) for row in cur.fetchall()]
