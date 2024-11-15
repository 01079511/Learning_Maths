from gm.api import *
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

# 定义常量
PRICE_TOLERANCE = 0.005  # 价格相等的容差
MIN_SHADOW = 0.01       # 影线最小长度
MAX_SHADOW = 0.03       # 影线最大长度

def is_price_equal(price1, price2):
    """判断两个价格是否相等(考虑精度)"""
    return abs(price1 - price2) <= PRICE_TOLERANCE

def is_t_pattern(row):
    """判断是否为T字或倒T字形态"""
    open_price = row['open']
    close_price = row['close']
    high_price = row['high']
    low_price = row['low']
    
    # 检查开盘价和收盘价是否相等
    if not is_price_equal(open_price, close_price):
        return None
    
    # T字形态：开盘=收盘=最低价，有合适长度的上影线
    if (is_price_equal(open_price, low_price) and 
        high_price > open_price):
        shadow_length = high_price - open_price
        if MIN_SHADOW - PRICE_TOLERANCE <= shadow_length <= MAX_SHADOW + PRICE_TOLERANCE:
            return 'T字形'
    
    # 倒T字形态：开盘=收盘=最高价，有合适长度的下影线
    if (is_price_equal(open_price, high_price) and 
        low_price < open_price):
        shadow_length = open_price - low_price
        if MIN_SHADOW - PRICE_TOLERANCE <= shadow_length <= MAX_SHADOW + PRICE_TOLERANCE:
            return '倒T字形'
    
    return None

def analyze_t_patterns(symbol='SHSE.600622', months=5):
    """分析T字形态并生成统计报告"""
    try:
        # 计算日期范围
        current_date = datetime.now()
        start_date = current_date - relativedelta(months=months)
        end_date = current_date
        
        # 获取1分钟K线数据
        data = history(symbol=symbol,
                      frequency='1m',
                      start_time=start_date,
                      end_time=end_date,
                      fields='eob,open,high,low,close,volume',
                      df=True)
        
        if data.empty:
            print("未获取到数据")
            return None
                
        # 获取股票名称    
        stock_info = get_instrumentinfos(symbols=symbol, df=True)
        stock_name = stock_info.iloc[0]['sec_name'] if not stock_info.empty else symbol
        
        # 添加日期时间列
        data['date'] = pd.to_datetime(data['eob']).dt.date
        data['time'] = pd.to_datetime(data['eob']).dt.strftime('%H:%M:%S')
        
        # 判断形态
        data['pattern_type'] = data.apply(is_t_pattern, axis=1)
        data['is_t_pattern'] = data['pattern_type'].notna()
        
        # 获取日线数据（移除不可用的 'turnover_ratio' 字段）
        daily_data = history(symbol=symbol,
                            frequency='1d',
                            start_time=start_date,
                            end_time=end_date,
                            fields='eob,open,close,high,low,amount',
                            adjust=ADJUST_PREV,
                            df=True)
        
        if daily_data.empty:
            print("未获取到日线数据")
            return None

        # 计算振幅 (高-低)/昨日收盘价
        daily_data['prev_close'] = daily_data['close'].shift(1)
        daily_data['amplitude'] = (daily_data['high'] - daily_data['low']) / daily_data['prev_close']
        daily_data['amplitude'] = daily_data['amplitude'].fillna(0)
        daily_data['date'] = pd.to_datetime(daily_data['eob']).dt.date

        # 打印日线数据的列名，确认可用字段
        # print(daily_data.columns)

        # 生成汇总统计
        generate_summary_csv(data, daily_data, stock_name, start_date.date(), end_date.date())
        
        # 生成最新一天详细数据
        generate_detail_csv(data, stock_name)
        
        print(f"已完成{stock_name}的T字形态分析")
        return True
            
    except Exception as e:
        print(f"分析过程错误: {e}")
        return None

def generate_summary_csv(data, daily_data, stock_name, start_date, end_date):
    """生成汇总统计CSV"""
    try:
        # 按天统计
        daily_stats = data.groupby('date').agg({
            'is_t_pattern': ['count', 'sum']
        }).reset_index()
        
        if daily_stats.empty:
            print("无统计数据")
            return
                
        # 重命名列
        daily_stats.columns = ['日期', '当日1分钟K线数', 'T字形态数']
        daily_stats['占比'] = (daily_stats['T字形态数'] / daily_stats['当日1分钟K线数'] * 100).round(2)
        daily_stats.insert(0, '股票名称', stock_name)
        
        # 创建 daily_info 的副本
        daily_info = daily_data[['date', 'open', 'close', 'amount', 'amplitude']].copy()

        # 重命名列
        daily_info.rename(columns={
        'open': '开盘价',
        'close': '收盘价',
        'amount': '成交额',
        'amplitude': '振幅'
        }, inplace=True)

        # 将 daily_info 与 daily_stats 合并
        daily_stats = pd.merge(daily_stats, daily_info, left_on='日期', right_on='date', how='left')
        daily_stats.drop(columns=['date'], inplace=True)

        # 指定完整文件路径
        filename = os.path.join(os.getcwd(), f'T形态汇总统计_{start_date}_{end_date}.csv')
        
        # 保存文件
        daily_stats.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"已生成汇总文件: {filename}")
            
    except Exception as e:
        print(f"生成汇总统计错误: {e}")

def generate_detail_csv(data, stock_name):
    """生成最近一天详细数据CSV"""
    try:
        # 获取最新日期
        latest_date = data['date'].max()
        
        # 筛选最新日期且为T字形态的数据
        latest_data = data[
            (data['date'] == latest_date) & 
            (data['pattern_type'].notna())
        ][['time', 'pattern_type', 'open', 'high', 'low', 'close']]
        
        if latest_data.empty:
            print("无详细数据")
            return
                
        # 指定完整文件路径
        filename = os.path.join(os.getcwd(), f'T形态详细数据_{latest_date}.csv')
        
        # 保存文件
        latest_data.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"已生成详细文件: {filename}")
        
    except Exception as e:
        print(f"生成详细数据错误: {e}")

def init(context):
    """策略初始化函数"""
    try:
        context.symbol = 'SZSE.002009'
        analyze_t_patterns(context.symbol)
    except Exception as e:
        print(f"初始化错误: {e}")

if __name__ == '__main__':
    run(strategy_id='strategy_id',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='{{token}}',
        backtest_start_time='2024-06-01 08:30:00',
        backtest_end_time='2024-11-15 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000)