from gm.api import *

def init(context,
        symbol = 'SZSE.002009', 
        backtest_start_time='2024-06-01',
        backtest_end_time='2024-11-15'):

    # 获取日线数据
    daily_data = get_history_symbol(symbol=symbol, 
                        start_date=backtest_start_time,
                        end_date=backtest_end_time,
                        df=True)
    
    # 打印可用字段名，用于调试
    print("可用字段:", daily_data.columns.tolist())
    
    # 打印数据
    if not daily_data.empty:
        print("\n天奇股份(002009)换手率数据:")
        print("日期\t\t换手率(%)")
        print("-" * 30)
        for _, row in daily_data.iterrows():
            print(f"{row['trade_date'].strftime('%Y-%m-%d')}\t{row['turn_rate']:.2f}")
    else:
        print("未获取到数据")

if __name__ == '__main__':
    run(strategy_id='strategy_id',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='{{token}}',
        backtest_start_time='2024-06-01 08:00:00',
        backtest_end_time='2024-11-15 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000)