from gm.api import *
import pandas as pd
import numpy as np

def init(context):
    # 设置参数
    context.symbol = 'SZSE.300353'
    context.frequency = '30m'
    context.start_date = '2024-10-31'
    context.end_date = '2024-11-02'
    context.period = 10
    context.multiplier = 3.0

    # 获取历史数据
    data = history_n(symbol=context.symbol,
                     frequency=context.frequency,
                     count=1000,
                     end_time=context.end_date + ' 16:00:00',
                     fields='eob,open,high,low,close',
                     adjust=ADJUST_NONE,
                     df=True)

    if data.empty:
        print("指定日期范围内没有数据")
        return

    # 筛选日期范围内的数据
    data = data[(data['eob'] >= context.start_date) & (data['eob'] <= context.end_date)]
    data.reset_index(drop=True, inplace=True)

    # 计算 ATR
    data['prev_close'] = data['close'].shift(1)
    data['tr'] = data[['high', 'prev_close']].max(axis=1) - data[['low', 'prev_close']].min(axis=1)
    data['atr'] = data['tr'].rolling(window=context.period).mean()

    # 计算基础上轨和下轨
    data['hl2'] = (data['high'] + data['low']) / 2
    data['basic_upperband'] = data['hl2'] + (context.multiplier * data['atr'])
    data['basic_lowerband'] = data['hl2'] - (context.multiplier * data['atr'])

    # 初始化 final_upperband 和 final_lowerband 列
    data['final_upperband'] = data['basic_upperband']
    data['final_lowerband'] = data['basic_lowerband']

    # 初始化趋势方向和 Supertrend
    data['trend'] = 1
    data['supertrend'] = np.nan

    for i in range(context.period, len(data)):
        # 更新 final_upperband 和 final_lowerband
        if data['close'].iloc[i - 1] <= data['final_upperband'].iloc[i - 1]:
            data['final_upperband'].iloc[i] = min(data['basic_upperband'].iloc[i], data['final_upperband'].iloc[i - 1])
        else:
            data['final_upperband'].iloc[i] = data['basic_upperband'].iloc[i]

        if data['close'].iloc[i - 1] >= data['final_lowerband'].iloc[i - 1]:
            data['final_lowerband'].iloc[i] = max(data['basic_lowerband'].iloc[i], data['final_lowerband'].iloc[i - 1])
        else:
            data['final_lowerband'].iloc[i] = data['basic_lowerband'].iloc[i]

        # 确定趋势方向
        if data['close'].iloc[i] > data['final_upperband'].iloc[i]:
            data['trend'].iloc[i] = 1
        elif data['close'].iloc[i] < data['final_lowerband'].iloc[i]:
            data['trend'].iloc[i] = -1
        else:
            data['trend'].iloc[i] = data['trend'].iloc[i - 1]

        # 根据趋势方向设置 Supertrend 值
        if data['trend'].iloc[i] == 1:
            data['supertrend'].iloc[i] = data['final_lowerband'].iloc[i]
        else:
            data['supertrend'].iloc[i] = data['final_upperband'].iloc[i]

    # 找出买入信号时间点（趋势从 -1 转为 1）
    buy_signals = data[(data['trend'] == 1) & (data['trend'].shift(1) == -1)]
    buy_times = buy_signals['eob']

    print('买入信号时间点:')
    for time in buy_times:
        print(time)

if __name__ == '__main__':
    run(strategy_id='',
        filename='test.py',
        mode=MODE_BACKTEST,
        token='{{token}}',  # 请将 YOUR_TOKEN 替换为您的实际 token
        backtest_start_time='2024-10-31 09:30:00',
        backtest_end_time='2024-11-02 15:00:00',
        backtest_initial_cash=1000000,
        backtest_adjust=ADJUST_NONE,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)