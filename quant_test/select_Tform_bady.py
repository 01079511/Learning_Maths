from gm.api import *
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

# 形态判断参数
MIN_SHADOW = 0.01  # 影线最小长度
MAX_SHADOW = 0.03  # 影线最大长度
PRICE_TOLERANCE = 0.001  # 价格相等的容差

def is_t_pattern(row):
    """判断是否为T字或倒T字形态"""
    try:
        open_price = row['open']
        close_price = row['close']
        high_price = row['high']
        low_price = row['low']
        
        # 检查开盘价和收盘价是否相等（允许小误差）
        if abs(open_price - close_price) > PRICE_TOLERANCE:
            return None
            
        # T字形态判断
        if (abs(open_price - low_price) <= PRICE_TOLERANCE and  # 开盘=收盘=最低价
            high_price > open_price):  # 有上影线
            shadow_length = high_price - open_price
            if MIN_SHADOW <= shadow_length <= MAX_SHADOW:
                return 'T字形'
                
        # 倒T字形态判断
        if (abs(open_price - high_price) <= PRICE_TOLERANCE and  # 开盘=收盘=最高价
            low_price < open_price):  # 有下影线
            shadow_length = open_price - low_price
            if MIN_SHADOW <= shadow_length <= MAX_SHADOW:
                return '倒T字形'
                
        return None
        
    except Exception as e:
        print(f"形态判断错误: {e}")
        return None

def analyze_t_patterns(symbol='SZSE.300353', months=5):
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
        data['time'] = pd.to_datetime(data['eob']).dt.time
        
        # 判断形态
        data['pattern_type'] = data.apply(is_t_pattern, axis=1)
        data['is_t_pattern'] = data['pattern_type'].notna()
        
        # 生成汇总统计
        generate_summary_csv(data, stock_name, start_date.date(), end_date.date())
        
        # 生成最新一天详细数据
        generate_detail_csv(data, stock_name)
        
        print(f"已完成{stock_name}的T字形态分析")
        return True
        
    except Exception as e:
        print(f"分析过程错误: {e}")
        return None

def generate_summary_csv(data, stock_name, start_date, end_date):
    """生成汇总统计CSV"""
    try:
        # 按天统计
        daily_stats = data.groupby('date').agg({
            'is_t_pattern': ['count', 'sum']
        }).reset_index()
        
        # 重命名列
        daily_stats.columns = ['日期', '当日1分钟K线数', 'T字型态数']
        daily_stats['占比'] = (daily_stats['T字型态数'] / daily_stats['当日1分钟K线数'] * 100).round(2)
        
        # 添加股票信息
        daily_stats.insert(0, '股票名称', stock_name)
        
        # 保存文件
        filename = f'T形态汇总统计_{start_date}_{end_date}.csv'
        daily_stats.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"已生成汇总统计: {filename}")
        
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
        
        # 保存文件
        filename = f'T形态详细数据_{latest_date}.csv'
        latest_data.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"已生成详细数据: {filename}")
        
    except Exception as e:
        print(f"生成详细数据错误: {e}")

def init(context):
    """策略初始化函数"""
    # 设置默认股票
    context.symbol = 'SZSE.300353'
    analyze_t_patterns(context.symbol)

if __name__ == '__main__':
    run(strategy_id='strategy_id',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='{{token}}',
        backtest_start_time='2023-01-23 08:00:00',
        backtest_end_time='2024-01-23 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000)