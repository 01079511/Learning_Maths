from gm.api import *
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

# 常量定义
PRICE_TOLERANCE = 0.005  # 价格相等的容差
MIN_SHADOW = 0.01       # 影线最小长度
MAX_SHADOW = 0.03       # 影线最大长度
PREDICT_WINDOW = 5      # 预测窗口(天数)
MIN_SAMPLES = 120       # 最小样本量

def is_price_equal(price1, price2):
    """判断两个价格是否相等(考虑精度)"""
    return abs(price1 - price2) <= PRICE_TOLERANCE

def is_t_pattern(row):
    """判断是否为T字或倒T字形态"""
    try:
        open_price = float(row['open'])
        close_price = float(row['close'])
        high_price = float(row['high'])
        low_price = float(row['low'])
        
        # 检查开盘价和收盘价是否相等
        if not is_price_equal(open_price, close_price):
            return None
            
        # T字形态：开盘=收盘=最低价，有合适长度的上影线
        if (is_price_equal(open_price, low_price) and high_price > open_price):
            shadow_length = high_price - open_price
            if MIN_SHADOW <= shadow_length <= MAX_SHADOW:
                return 'T字形'
                
        # 倒T字形态：开盘=收盘=最高价，有合适长度的下影线
        if (is_price_equal(open_price, high_price) and low_price < open_price):
            shadow_length = open_price - low_price
            if MIN_SHADOW <= shadow_length <= MAX_SHADOW:
                return '倒T字形'
                
        return None
    except Exception as e:
        print(f"判断T型形态时出错: {e}")
        return None

def analyze_t_patterns(symbol):
    """分析T字形态并研究其与价格变动的关系"""
    try:
        # 获取历史数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        # 获取1分钟K线数据
        data = history(symbol=symbol,
                      frequency='1m',
                      start_time=start_date,
                      end_time=end_date,
                      fields='eob,open,high,low,close,volume',
                      df=True)
        
        if data.empty:
            print(f"{symbol} 未获取到数据")
            return None
            
        # 获取股票名称    
        stock_info = get_instrumentinfos(symbols=symbol, df=True)
        stock_name = stock_info.iloc[0]['sec_name'] if not stock_info.empty else symbol
        
        # 添加日期时间列
        data['date'] = pd.to_datetime(data['eob']).dt.date
        data['time'] = pd.to_datetime(data['eob']).dt.strftime('%H:%M:%S')
        
        # 判断T型形态
        data['pattern_type'] = data.apply(is_t_pattern, axis=1)
        data['is_t_pattern'] = data['pattern_type'].notna()
        
        # 统计分析
        daily_stats = calculate_daily_stats(data)
        
        # 预测分析
        prediction_results = analyze_pattern_impact(data, daily_stats)
        
        # 生成报告
        generate_reports(daily_stats, prediction_results, stock_name, start_date, end_date)
        
        return True
        
    except Exception as e:
        print(f"分析过程错误: {e}")
        return None

def calculate_daily_stats(data):
    """计算每日T型形态统计"""
    daily_stats = data.groupby('date').agg({
        'is_t_pattern': ['count', 'sum']
    }).reset_index()
    
    daily_stats.columns = ['date', 'total_patterns', 't_patterns']
    daily_stats['t_pattern_ratio'] = daily_stats['t_patterns'] / daily_stats['total_patterns']
    
    return daily_stats

def analyze_pattern_impact(data, daily_stats):
    """分析T型形态对后续价格的影响"""
    # 计算日级别收益率
    daily_prices = data.groupby('date')['close'].last().reset_index()
    daily_prices['return'] = daily_prices['close'].pct_change(PREDICT_WINDOW)
    
    # 合并T型形态统计和收益率
    analysis_df = pd.merge(daily_stats, daily_prices[['date', 'return']], on='date')
    
    # 相关性分析
    correlation = analysis_df['t_pattern_ratio'].corr(analysis_df['return'])
    
    # 分组分析
    analysis_df['t_pattern_quantile'] = pd.qcut(analysis_df['t_pattern_ratio'], 
                                              5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    group_analysis = analysis_df.groupby('t_pattern_quantile')['return'].mean()
    
    return {
        'correlation': correlation,
        'group_analysis': group_analysis,
        'daily_data': analysis_df
    }

def generate_reports(daily_stats, prediction_results, stock_name, start_date, end_date):
    """生成分析报告"""
    # 生成汇总统计CSV
    daily_stats['stock_name'] = stock_name
    filename = f'T形态汇总统计_{start_date.date()}_{end_date.date()}.csv'
    daily_stats.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"已生成汇总统计: {filename}")
    
    # 生成分析报告
    report_filename = f'T形态分析报告_{stock_name}.txt'
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(f"T型形态分析报告 - {stock_name}\n")
        f.write(f"分析期间: {start_date.date()} 至 {end_date.date()}\n\n")
        f.write(f"1. T型形态与{PREDICT_WINDOW}日收益率相关性: {prediction_results['correlation']:.4f}\n\n")
        f.write("2. 分组分析结果:\n")
        f.write(prediction_results['group_analysis'].to_string())
        
    print(f"已生成分析报告: {report_filename}")

def init(context):
    try:
        symbols_list = ['SZSE.300353', 'SHSE.600101', 'SHSE.600622']
        for symbol in symbols_list:
            analyze_t_patterns(symbol)
    except Exception as e:
        print(f"初始化错误: {e}")


if __name__ == '__main__':
    run(strategy_id='strategy_id',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='{{token}}',
        backtest_start_time='2024-06-01 08:00:00',
        backtest_end_time='2024-11-13 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000)