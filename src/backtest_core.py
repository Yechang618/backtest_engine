# /data/cye_temp/workspace/backtest_engine/src/backtest_core.py
import pandas as pd
import numpy as np
from typing import List, Dict
import logging
import os

logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self, initial_capital: float, commission_rate: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.positions = {}  # {code: {'shares': int}}
        self.trades = []
        self.last_known_prices = {}
        
        # 🔑 新增：成本追踪与胜率统计
        self.cost_basis = {}  # {code: float} 记录加权平均买入成本
        self.trade_stats = {'wins': 0, 'losses': 0, 'total_closed': 0}

    def update_daily(self, date: pd.Timestamp, price_dict: Dict[str, float]) -> float:
        pos_value = 0.0
        for code, pos in self.positions.items():
            price = price_dict.get(code, 0.0)
            if price > 1e-6:
                self.last_known_prices[code] = price
            else:
                price = self.last_known_prices.get(code, 0.0)
            pos_value += pos['shares'] * price
        return self.cash + pos_value

    def rebalance(self, date: pd.Timestamp, top50: List[str], price_dict: Dict[str, float]):
        if not top50:
            return

        pos_value = sum(pos['shares'] * price_dict.get(c, self.last_known_prices.get(c, 0.0)) for c, pos in self.positions.items())
        total_nav = self.cash + pos_value
        if total_nav <= 1e-3 or len(top50) == 0:
            return

        target_weight = 1.0 / len(top50)
        target_value_per_stock = total_nav * target_weight

        trades_plan = []
        target_codes = set(top50)

        for code in top50:
            price = price_dict.get(code, 0.0)
            if price <= 1e-6: continue
            target_shares = int(target_value_per_stock / price / 100) * 100
            current_shares = self.positions.get(code, {}).get('shares', 0)
            delta = target_shares - current_shares
            if delta != 0:
                trades_plan.append({'code': code, 'price': price, 'delta': delta, 'type': 'BUY' if delta > 0 else 'SELL'})

        for code in list(self.positions.keys()):
            if code not in target_codes:
                current_shares = self.positions[code]['shares']
                price = price_dict.get(code, 0.0)
                if current_shares > 0 and price > 1e-6:
                    trades_plan.append({'code': code, 'price': price, 'delta': -current_shares, 'type': 'SELL'})

        trades_plan.sort(key=lambda x: 0 if x['type'] == 'SELL' else 1)

        for t in trades_plan:
            code, price, delta, action = t['code'], t['price'], t['delta'], t['type']
            
            if action == 'SELL':
                shares = abs(delta)
                proceeds = shares * price
                fee = proceeds * (self.commission_rate + 0.0000)
                self.cash += proceeds - fee
                
                # 🔑 核心新增：统计单笔平仓胜率
                cost_price = self.cost_basis.get(code, price)
                if price > cost_price:
                    self.trade_stats['wins'] += 1
                else:
                    self.trade_stats['losses'] += 1
                self.trade_stats['total_closed'] += 1
                
                if code in self.positions:
                    self.positions[code]['shares'] -= shares
                    if self.positions[code]['shares'] <= 0:
                        del self.positions[code]
                        if code in self.cost_basis:
                            del self.cost_basis[code]
                            
                self.trades.append({'date': date, 'code': code, 'action': 'SELL', 'shares': shares, 'price': price, 'fee': fee})

            elif action == 'BUY':
                shares = delta
                cost = shares * price
                fee = cost * self.commission_rate
                
                if self.cash >= cost + fee:
                    self.cash -= (cost + fee)
                    # 🔑 核心新增：更新加权平均买入成本
                    current_shares = self.positions.get(code, {}).get('shares', 0)
                    current_cost = self.cost_basis.get(code, 0.0)
                    total_shares = current_shares + shares
                    self.cost_basis[code] = (current_cost * current_shares + price * shares) / total_shares
                    
                    self.positions[code] = {'shares': total_shares}
                    self.trades.append({'date': date, 'code': code, 'action': 'BUY', 'shares': shares, 'price': price, 'fee': fee})
                else:
                    max_shares = int(self.cash / (price * (1 + self.commission_rate)) / 100) * 100
                    if max_shares >= 100:
                        actual_cost = max_shares * price
                        actual_fee = actual_cost * self.commission_rate
                        self.cash -= (actual_cost + actual_fee)
                        
                        # 🔑 核心新增：降级买入同样更新成本
                        current_shares = self.positions.get(code, {}).get('shares', 0)
                        current_cost = self.cost_basis.get(code, 0.0)
                        total_shares = current_shares + max_shares
                        self.cost_basis[code] = (current_cost * current_shares + price * max_shares) / total_shares
                        
                        self.positions[code] = {'shares': total_shares}
                        self.trades.append({'date': date, 'code': code, 'action': 'BUY', 'shares': max_shares, 'price': price, 'fee': actual_fee})

    def buy_universe_once(self, date, stock_list: List[str], price_dict: Dict[str, float]):
        if not stock_list or self.cash <= 1e-3:
            return
        n = len(stock_list)
        budget = self.cash / n
        for code in stock_list:
            price = price_dict.get(code, 0.0)
            if price <= 1e-6: continue
            shares = int(budget / price / 100) * 100
            if shares <= 0: continue
            cost = shares * price
            fee = cost * self.commission_rate
            if self.cash >= cost + fee:
                self.cash -= (cost + fee)
                # 🔑 新增：基线建仓也记录成本
                self.cost_basis[code] = price
                self.positions[code] = {'shares': shares}
                self.trades.append({'date': date, 'code': code, 'action': 'BUY', 'shares': shares, 'price': price, 'fee': fee})

    def save_logs(self, model_name: str, log_dir: str):
        if self.trades:
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, f"trades_{model_name}.csv")
            pd.DataFrame(self.trades).to_csv(path, index=False)
            logger.info(f"📝 {model_name} 交易明细已保存: {path}")