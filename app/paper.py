from __future__ import annotations
from .models import PaperPosition, TradeRecord

class PaperBroker:
    def __init__(self, starting_cash: float):
        self.starting_cash=starting_cash; self.cash=starting_cash; self.positions: dict[str,PaperPosition]={}; self.realized_pnl=0.0; self.trades: list[TradeRecord]=[]; self.kill_switch=False
    def open(self,ticker:str,side:str,quantity:int,price:float,reason:str)->TradeRecord:
        if ticker in self.positions: raise ValueError('Position already exists for market')
        cost=quantity*price
        if cost>self.cash+1e-9: raise ValueError('Insufficient paper cash')
        self.cash-=cost; self.positions[ticker]=PaperPosition(ticker=ticker,side=side,quantity=quantity,entry_price=price,peak_mark=price); trade=TradeRecord(ticker=ticker,side=side,action='open',quantity=quantity,price=price,reason=reason); self.trades.append(trade); return trade
    def close(self,ticker:str,price:float,reason:str)->TradeRecord:
        pos=self.positions.pop(ticker); proceeds=pos.quantity*price; cost=pos.quantity*pos.entry_price; pnl=proceeds-cost; self.cash+=proceeds; self.realized_pnl+=pnl; trade=TradeRecord(ticker=ticker,side=pos.side,action='close',quantity=pos.quantity,price=price,pnl=pnl,reason=reason); self.trades.append(trade); return trade
    def settle(self,ticker:str,result:str,reason:str='Kalshi market settled')->TradeRecord:
        pos=self.positions.pop(ticker); price=1.0 if result.lower()==pos.side else 0.0; proceeds=pos.quantity*price; cost=pos.quantity*pos.entry_price; pnl=proceeds-cost; self.cash+=proceeds; self.realized_pnl+=pnl; trade=TradeRecord(ticker=ticker,side=pos.side,action='settle',quantity=pos.quantity,price=price,pnl=pnl,reason=reason); self.trades.append(trade); return trade
    def mark_equity(self,marks:dict[str,float])->float:
        value=self.cash
        for ticker,pos in self.positions.items():
            mark=marks.get(ticker,pos.entry_price); pos.peak_mark=max(pos.peak_mark,mark); value+=pos.quantity*mark
        return value
