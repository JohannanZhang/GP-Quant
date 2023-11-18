# Author: Allen Chi <allenchi.cn>
# License: BSD 3 clause
# Create Date 2020/07/31
from gplearn import _program
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

from scipy import stats
from tqdm import tqdm


class _Account:
    def __init__(self,
                 t,
                 cash=1000000,
                 fee_rate=0,
                 skid=0,
                 stop_loss=-2,
                 profit_taking=2,
                 stop_time=2.5):
        self.cash = cash
        self.fee_rate = fee_rate
        self.skid = skid
        self.stop_loss = stop_loss
        self.profit_taking = profit_taking
        self.stop_time = stop_time

        self.equity = self.cash
        self.unused_cash = self.cash
        self.multiple = 300
        self.margin_rate = 0.1
        self.lng_pos = 0
        self.sht_pos = 0
        self.margin = 0
        self.open_order_id = []
        self.balance = 0

        self.wallet_df = pd.DataFrame(columns=['equity', 'cash', 'unused_cash', 'lng_pos', 'sht_pos',
                                               'margin', 'time', 'price', 'open_order_id'])
        self.order_df = pd.DataFrame(columns=['open_time', 'close_time', 'open_price', 'close_price',
                                              'direction', 'amount', 'profit', 'margin',
                                              'fee', 'state', 'close_type'])
        self.open_order_df = self.order_df.copy()
        self.return_ls = []

        self.wallet_df.loc[0] = [self.equity, self.cash, self.unused_cash, self.lng_pos, self.sht_pos,
                                 self.margin, t, np.nan, self.open_order_id.copy()]

    def open_order(self, price, t, direction, amount, change=False):
        price = price * (1 + self.skid * direction)
        value = self.multiple * price * amount
        if change:
            fee = 0
        else:
            fee = value * self.fee_rate
        margin = value * self.margin_rate
        if self.unused_cash > fee + margin:  # 可以开仓
            _ = datetime.strftime(t, '%Y%m%d%H%M%S')
            if self.order_df.shape[0] > 0:
                order_id = _ + '%d' % (sum(i[:-1] == _ for i in self.order_df.index[-2:]) + 1)
            else:
                order_id = _ + '1'
            self.order_df.loc[order_id] = [t, np.nan, price, np.nan,
                                           direction, amount, 0, margin,
                                           fee, True, np.nan]
            self.cash -= fee
            self.margin += margin
            self.open_order_id.append(order_id)
            if direction == 1:
                self.lng_pos += amount
            else:
                self.sht_pos += amount
        else:  # 可用现金不足，无法开仓
            pass

    def close_order(self, price, t, order_id, close_type, change=False):
        open_price, direction, amount, margin, open_fee, state = self.order_df.loc[
            order_id, ['open_price', 'direction', 'amount', 'margin', 'fee', 'state']
        ].values.tolist()

        if state:  # 可以平仓
            price = price * (1 - self.skid * direction)
            value = self.multiple * price * amount
            if change:
                fee = 0
            else:
                fee = value * self.fee_rate
            profit = (price - open_price) * amount * direction * self.multiple
            self.order_df.loc[order_id, ['close_time', 'close_price', 'profit', 'fee',
                                         'state', 'close_type']] = \
                [t, price, profit, fee + open_fee, False, close_type]
            self.cash -= fee
            self.cash += profit
            self.margin -= margin
            self.open_order_id.remove(order_id)
            if direction == 1:
                self.lng_pos -= amount
            else:
                self.sht_pos -= amount
        else:  # 订单状态错误，无法平仓
            pass

    def update_wallet(self, bid, ask, t):  # 接收下一tick数据前执行
        self.balance = self.lng_pos - self.sht_pos
        self.open_order_df = self.order_df.loc[self.open_order_id]
        self.equity = self.cash - ((self.open_order_df['direction'] * self.open_order_df['amount'] *
                                    self.open_order_df['open_price']).sum() -
                                   bid * self.lng_pos + ask * self.sht_pos) * self.multiple
        self.unused_cash = self.cash - self.margin
        self.wallet_df.loc[len(self.wallet_df)] = [self.equity, self.cash, self.unused_cash, self.lng_pos, self.sht_pos,
                                                   self.margin, t, (bid + ask) / 2, self.open_order_id.copy()]

    def max_drawdown(self):
        i = int(np.argmax((np.maximum.accumulate(self.return_ls) - self.return_ls) /
                          np.maximum.accumulate(self.return_ls)))
        if i == 0:
            return 0
        j = int(np.argmax(self.return_ls[:i]))  # 开始位置

        return (self.return_ls[j] - self.return_ls[i]) / self.return_ls[j]

    def sharpe(self):
        mean = (np.mean(self.return_ls) + 1)
        std = np.std(self.return_ls)
        sharpe = mean / std * np.sqrt(252 * 4 * 60 * 60 * 2)
        return std, sharpe


class _Fitness:
    def __init__(self, signal, bid, ask, dttm_index):
        self.dttm_index = dttm_index.tolist()
        self.data: pd.DataFrame = pd.concat([bid, ask, dttm_index], axis=1)
        signal[np.isinf(signal)] = np.nan
        self.data['signal'] = signal
        self.data.columns = ['bid', 'ask', 'dttm', 'signal']
        self.data = self.data.set_index('dttm')
        self.sig_rk = self.data['signal'].rolling(1000, min_periods=1).apply(
            lambda x: stats.percentileofscore(x, x[-1])
        )
        self.bid = self.data['bid']
        self.ask = self.data['ask']
        self.account = _Account(t=dttm_index[0])

    def close_stop(self, bid, ask, t):
        if self.account.balance > 0:
            profit = bid - self.account.open_order_df.open_price
        elif self.account.balance < 0:
            profit = ask - self.account.open_order_df.open_price
        else:
            return {}

        close_order_1 = profit[profit >= self.account.profit_taking].index.tolist()
        close_order_2 = profit[profit <= self.account.stop_loss].index.tolist()

        time_delay = t - timedelta(seconds=self.account.stop_time)
        close_order_3 = self.account.open_order_df.loc[
            self.account.open_order_df['open_time'] <= time_delay].index.tolist()

        return {
            **{order: '止盈' for order in close_order_1},
            **{order: '止损' for order in close_order_2},
            **{order: '超时' for order in set(close_order_3) - set(close_order_1) - set(close_order_2)}
        }

    def handle_tick(self, t):
        bid = self.bid[t]
        ask = self.ask[t]

        # 开平仓
        signal_rk = self.sig_rk[t]

        if self.account.balance == 0:
            if signal_rk >= 90:
                self.account.open_order(ask, t, 1, 1)  # 开多仓
            elif signal_rk <= 10:
                self.account.open_order(bid, t, -1, 1)  # 开空仓

        elif self.account.balance > 0:
            if signal_rk >= 90:
                close_order = self.close_stop(bid, ask, t)  # 判断是否止盈/止损/超时
                if close_order:
                    order = list(close_order.keys())[0]
                    self.account.close_order(
                        price=bid, t=t, order_id=order, close_type=close_order.get(order), change=True)
                    self.account.open_order(bid, t, 1, 1, change=True)  # 多换（无手续费）
                    for order in list(close_order.keys())[1:]:
                        self.account.close_order(
                            price=bid, t=t, order_id=order, close_type=close_order.get(order))
                else:
                    self.account.open_order(ask, t, 1, 1)  # 加多
        elif self.account.balance < 0:
            if signal_rk <= 10:
                close_order = self.close_stop(bid, ask, t)  # 判断是否止盈/止损/超时
                if close_order:
                    order = list(close_order.keys())[0]
                    self.account.close_order(
                        price=ask, t=t, order_id=order, close_type=close_order.get(order), change=True)
                    self.account.open_order(ask, t, -1, 1, change=True)  # 空换（无手续费）
                    for order in list(close_order.keys())[1:]:
                        self.account.close_order(
                            price=ask, t=t, order_id=order, close_type=close_order.get(order))
                else:
                    self.account.open_order(bid, t, -1, 1)  # 加空
        else:  # We never get here
            pass

        # 更新账户信息
        self.account.update_wallet(bid, ask, t)

    def handel_day(self, date):
        # 每天 9:50-11:30 、13:00-14:55 交易
        dttm_ls = [dttm for dttm in self.dttm_index if dttm.date() == date]
        dttm_order = [dttm for dttm in dttm_ls if time(9, 50) < dttm.time() < time(14, 55)]
        for dttm in tqdm(dttm_order):
            self.handle_tick(dttm)

        # 每天 14:55 强行平仓
        if self.account.open_order_id:
            t = np.min([dttm for dttm in dttm_ls if dttm.time() >= time(14, 55)])
            bid, ask = self.bid[t], self.ask[t]
            for order_id in self.account.open_order_id:
                if self.account.open_order_df.loc[order_id, 'direction'] == 1:
                    self.account.close_order(price=bid, t=t, order_id=order_id, close_type='收盘')
                else:
                    self.account.close_order(price=ask, t=t, order_id=order_id, close_type='收盘')
            self.account.update_wallet(bid, ask, t)

    def fit(self):
        if self.data['signal'].isna().sum() / len(self.data) > 0.001:
            return np.nan
        else:
            date_ls = list(set(dttm.date() for dttm in self.dttm_index))
            date_ls.sort()
            for date in date_ls:
                self.handel_day(date)
