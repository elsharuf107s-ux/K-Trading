from __future__ import annotations
import json, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from .models import FeatureVector, Forecast, MarketSnapshot, TradeRecord

class Storage:
    def __init__(self,path:str):
        Path(path).parent.mkdir(parents=True,exist_ok=True); self.db=sqlite3.connect(path,check_same_thread=False); self.db.execute('PRAGMA journal_mode=WAL'); self.db.executescript('''
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT NOT NULL,ticker TEXT NOT NULL,btc_price REAL NOT NULL,
            market_probability REAL NOT NULL,yes_bid REAL,yes_ask REAL,no_bid REAL,no_ask REAL,seconds_remaining REAL,
            model_probability REAL,confidence REAL,reference_price REAL,reference_source TEXT,book_imbalance REAL,wave_score REAL,
            usable_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT NOT NULL,ticker TEXT NOT NULL,side TEXT NOT NULL,
            action TEXT NOT NULL,quantity INTEGER NOT NULL,price REAL NOT NULL,pnl REAL NOT NULL,reason TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS market_results (
            ticker TEXT PRIMARY KEY,result TEXT NOT NULL,settled_at TEXT NOT NULL
        );
        '''); self._migrate_observations(); self.db.commit()
    def _migrate_observations(self):
        columns={row[1] for row in self.db.execute('PRAGMA table_info(observations)').fetchall()}; additions={'reference_price':'REAL','reference_source':'TEXT','book_imbalance':'REAL','wave_score':'REAL'}
        for name,typ in additions.items():
            if name not in columns: self.db.execute(f'ALTER TABLE observations ADD COLUMN {name} {typ}')
    def save_observation(self,market:MarketSnapshot,features:FeatureVector,forecast:Forecast):
        payload={'market':market.model_dump(mode='json'),'features':features.model_dump(mode='json'),'forecast':forecast.model_dump(mode='json')}; self.db.execute('''INSERT INTO observations(created_at,ticker,btc_price,market_probability,yes_bid,yes_ask,no_bid,no_ask,seconds_remaining,model_probability,confidence,reference_price,reference_source,book_imbalance,wave_score,usable_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(market.captured_at.isoformat(),market.ticker,features.btc_price,features.market_probability,features.yes_bid,features.yes_ask,features.no_bid,features.no_ask,features.seconds_remaining,forecast.yes_probability,forecast.confidence,features.reference_price,features.reference_source,features.book_imbalance,features.wave_score,json.dumps(payload))); self.db.commit()
    def save_trade(self,trade:TradeRecord):
        self.db.execute('INSERT INTO trades(created_at,ticker,side,action,quantity,price,pnl,reason) VALUES(?,?,?,?,?,?,?,?)',(trade.created_at.isoformat(),trade.ticker,trade.side,trade.action,trade.quantity,trade.price,trade.pnl,trade.reason)); self.db.commit()
    def save_market_result(self,ticker:str,result:str,settled_at:str|None=None):
        result=result.lower()
        if result not in ('yes','no'): raise ValueError('Market result must be yes or no')
        settled_at=settled_at or datetime.now(timezone.utc).isoformat(); self.db.execute('INSERT INTO market_results(ticker,result,settled_at) VALUES(?,?,?) ON CONFLICT(ticker) DO UPDATE SET result=excluded.result,settled_at=excluded.settled_at',(ticker,result,settled_at)); self.db.commit()
    def unresolved_tickers(self,current_ticker:str='',limit:int=10)->list[str]:
        cur=self.db.execute('''SELECT DISTINCT o.ticker FROM observations o LEFT JOIN market_results r ON r.ticker=o.ticker WHERE r.ticker IS NULL AND o.ticker<>? ORDER BY o.id DESC LIMIT ?''',(current_ticker,limit)); return [row[0] for row in cur.fetchall()]
    def recent_trades(self,limit:int=50)->list[dict]:
        cur=self.db.execute('SELECT created_at,ticker,side,action,quantity,price,pnl,reason FROM trades ORDER BY id DESC LIMIT ?',(limit,)); keys=['created_at','ticker','side','action','quantity','price','pnl','reason']; return [dict(zip(keys,row)) for row in cur.fetchall()]
    def closed_trade_records(self,limit:int=20000)->list[dict]:
        cur=self.db.execute("SELECT created_at,ticker,side,action,quantity,price,pnl,reason FROM trades WHERE action IN ('close','settle') ORDER BY id DESC LIMIT ?",(limit,)); keys=['created_at','ticker','side','action','quantity','price','pnl','reason']; return [dict(zip(keys,row)) for row in cur.fetchall()]
    def recent_observations(self,limit:int=200)->list[dict]:
        cur=self.db.execute('SELECT created_at,ticker,market_probability,model_probability,reference_price,reference_source,book_imbalance,wave_score,seconds_remaining FROM observations ORDER BY id DESC LIMIT ?',(limit,)); keys=['created_at','ticker','market_probability','model_probability','reference_price','reference_source','book_imbalance','wave_score','seconds_remaining']; return [dict(zip(keys,row)) for row in cur.fetchall()]
    def labeled_observations(self,limit:int=100000)->list[dict]:
        cur=self.db.execute('''SELECT o.created_at,o.ticker,o.market_probability,o.model_probability,o.seconds_remaining,r.result,CASE WHEN r.result='yes' THEN 1.0 ELSE 0.0 END outcome_yes FROM observations o JOIN market_results r ON r.ticker=o.ticker ORDER BY o.id DESC LIMIT ?''',(limit,)); keys=['created_at','ticker','market_probability','model_probability','seconds_remaining','result','outcome_yes']; return [dict(zip(keys,row)) for row in cur.fetchall()]
    def last_labeled_observations(self,limit:int=20000)->list[dict]:
        cur=self.db.execute('''SELECT o.created_at,o.ticker,o.market_probability,o.model_probability,o.seconds_remaining,r.result,CASE WHEN r.result='yes' THEN 1.0 ELSE 0.0 END outcome_yes FROM observations o JOIN (SELECT ticker,MAX(id) max_id FROM observations GROUP BY ticker) x ON x.max_id=o.id JOIN market_results r ON r.ticker=o.ticker ORDER BY o.id DESC LIMIT ?''',(limit,)); keys=['created_at','ticker','market_probability','model_probability','seconds_remaining','result','outcome_yes']; return [dict(zip(keys,row)) for row in cur.fetchall()]
    def replay_window(self,ticker:str)->dict|None:
        cur=self.db.execute('SELECT created_at,usable_json FROM observations WHERE ticker=? ORDER BY id ASC',(ticker,)); rows=cur.fetchall()
        if not rows: return None
        result_row=self.db.execute('SELECT result,settled_at FROM market_results WHERE ticker=?',(ticker,)).fetchone()
        return {'ticker':ticker,'result':result_row[0] if result_row else None,'settled_at':result_row[1] if result_row else None,'frames':[{'created_at':created_at,**json.loads(payload)} for created_at,payload in rows]}
