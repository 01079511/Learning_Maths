from gm.api import *
import pandas as pd

def get_kline_data(symbol, date, times):
    # 获取当日所有1分钟K线数据
    data = history(symbol=symbol, frequency='1m', start_time=f"{date} 09:30:00", end_time=f"{date} 15:00:00",
                   fields='eob,open,high,low,close,volume', df=True)
    if data.empty:
        print("未获取到数据")
        return
    # 提取时间信息
    data['time'] = pd.to_datetime(data['eob']).dt.strftime('%H:%M:%S')
    # 筛选指定时间点的数据
    selected_data = data[data['time'].isin(times)]
    # 选择需要的列
    selected_data = selected_data[['time', 'open', 'high', 'low', 'close', 'volume']]
    # 保存为CSV文件
    filename = f"{symbol}_{date}_selected_times.csv"
    selected_data.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"已保存CSV文件：{filename}")

def init(context):
    """初始化函数"""
    symbol = 'SZSE.300353'
    date = '2024-11-13'
    check_times = [
        '09:48:00', '11:21:00', '11:22:00', '11:29:00',
        '13:17:00', '13:39:00', '13:40:00', '13:43:00',
        '13:46:00', '13:53:00', '13:56:00', '14:10:00',
        '14:19:00', '14:37:00', '14:43:00'
    ]
    get_kline_data(symbol, date, check_times)

if __name__ == '__main__':
    run(strategy_id='strategy_id',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='{{token}}',
        backtest_start_time='2024-11-13 09:30:00',
        backtest_end_time='2024-11-13 15:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000)