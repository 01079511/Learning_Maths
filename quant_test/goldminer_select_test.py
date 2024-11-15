from gm.api import *
import pandas as pd
from datetime import datetime, timedelta

# 配置常量
PRICE_TOLERANCE = 0.005  # 价格相等的容差
MIN_SHADOW = 0.01       # 影线最小长度
MAX_SHADOW = 0.03       # 影线最大长度

# 推荐标准阈值
T_PATTERN_THRESHOLD = 0.20      # T型K线总占比阈值
MORNING_T_THRESHOLD = 0.10      # 早盘T型K线占比阈值
CONTINUOUS_WINDOW = 15          # 连续性检测窗口(分钟)
MAX_CONTINUOUS_WINDOWS = 2      # 允许的最大连续窗口数

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

def analyze_continuous_patterns(data, window_size=CONTINUOUS_WINDOW):
    """分析连续T型K线"""
    windows = []
    current_window = []
    last_pattern_time = None
    
    for idx, row in data.iterrows():
        if row['is_t_pattern']:
            current_time = pd.to_datetime(row['eob'])
            
            if last_pattern_time is None:
                current_window = [current_time]
            else:
                time_diff = (current_time - last_pattern_time).total_seconds() / 60
                if time_diff <= window_size:
                    current_window.append(current_time)
                else:
                    if len(current_window) > 1:
                        windows.append(current_window)
                    current_window = [current_time]
            
            last_pattern_time = current_time
    
    if len(current_window) > 1:
        windows.append(current_window)
        
    return len(windows)  # 返回连续窗口数量

# 推荐标准阈值
T_PATTERN_THRESHOLD_L1 = 0.15    # Level 1的全天T型占比阈值(15%)
T_PATTERN_THRESHOLD_L2 = 0.20    # Level 2的全天T型占比阈值(20%)
MORNING_T_THRESHOLD_L1 = 0.08    # Level 1的早盘T型占比阈值(8%)
MORNING_T_THRESHOLD_L2 = 0.10    # Level 2的早盘T型占比阈值(10%)
CONTINUOUS_WINDOWS_L1 = 3        # Level 1的连续窗口阈值
CONTINUOUS_WINDOWS_L2 = 6        # Level 2的连续窗口阈值

def evaluate_stock(total_ratio, morning_ratio, continuous_windows):
    """评估股票是否推荐
    
    Level 1 (推荐使用):
    - 早盘T型占比 < 8%
    - 全天T型占比 < 15%
    - 连续性窗口 ≤ 3
    
    Level 2 (观察使用):
    - 早盘T型占比 8-10%
    - 全天T型占比 15-20%
    - 连续性窗口 ≤ 6
    
    Level 3 (不推荐使用):
    - 超出以上任一标准
    """
    # Level 3 判定（任一指标超标）
    if (total_ratio > T_PATTERN_THRESHOLD_L2 or 
        morning_ratio > MORNING_T_THRESHOLD_L2 or 
        continuous_windows > CONTINUOUS_WINDOWS_L2):
        return 3, "不推荐：指标超出可接受范围"
    
    # Level 1 判定（所有指标都在最优范围）
    if (total_ratio < T_PATTERN_THRESHOLD_L1 and 
        morning_ratio < MORNING_T_THRESHOLD_L1 and 
        continuous_windows <= CONTINUOUS_WINDOWS_L1):
        return 1, "推荐：所有指标都在理想范围内"
    
    # Level 2 判定（介于Level 1和Level 3之间）
    return 2, "观察：指标在临界范围内"

def analyze_t_patterns(context, symbol):
    """分析T字形态并生成统计报告"""
    try:
        # 获取最近5个交易日
        current_date = datetime.now()
        end_date = current_date
        start_date = current_date - timedelta(days=5)
        
        # 获取1分钟K线数据
        data = history(symbol=symbol,
                      frequency='60s',
                      start_time=start_date,
                      end_time=end_date,
                      fields='eob,open,high,low,close,volume',
                      df=True)
        
        if data.empty:
            print(f"{symbol} 未获取到数据")
            return None
            
        # 获取股票名称 (修改为使用get_symbol_infos)    
        stock_info = get_symbol_infos(
            sec_type1=1010,  # 股票类型
            symbols=symbol,
            df=True
        )
        stock_name = stock_info.iloc[0]['sec_name'] if not stock_info.empty else symbol
        
        # 添加日期时间列
        data['date'] = pd.to_datetime(data['eob']).dt.date
        data['time'] = pd.to_datetime(data['eob']).dt.strftime('%H:%M:%S')
        
        # 判断形态
        data['pattern_type'] = data.apply(is_t_pattern, axis=1)
        data['is_t_pattern'] = data['pattern_type'].notna()
        
        # 计算连续性指标
        continuous_windows = analyze_continuous_patterns(data)
        
        # 生成汇总统计
        generate_summary_csv(data, stock_name, start_date.date(), end_date.date(), continuous_windows)
        
        # 生成最新一天详细数据
        generate_detail_csv(data, stock_name)
        
        print(f"已完成{stock_name}的T字形态分析")
        return True
        
    except Exception as e:
        print(f"分析过程错误: {e}")
        return None

def generate_summary_csv(data, stock_name, start_date, end_date, continuous_windows):
    """生成汇总统计CSV"""
    try:
        # 按天统计
        daily_stats = []
        
        for date, group in data.groupby('date'):
            # 计算全天统计
            total_klines = len(group)
            total_t_patterns = group['is_t_pattern'].sum()
            total_ratio = (total_t_patterns / total_klines * 100) if total_klines > 0 else 0
            
            # 计算早盘统计（9:30-10:30）
            morning_data = group[group['time'].between('09:30:00', '10:30:00')]
            morning_klines = len(morning_data)
            morning_t_patterns = morning_data['is_t_pattern'].sum()
            morning_ratio = (morning_t_patterns / morning_klines * 100) if morning_klines > 0 else 0
            
            # 评估推荐等级
            level, reason = evaluate_stock(total_ratio, morning_ratio, continuous_windows)
            
            daily_stats.append({
                '股票名称': stock_name,
                '日期': date,
                '当日1分钟K线数': total_klines,
                'T字型态数': total_t_patterns,
                '全天占比': round(total_ratio, 2),
                '早盘占比': round(morning_ratio, 2),
                '连续窗口数': continuous_windows,
                '推荐等级': level,
                '评估结果': reason
            })
        
        if not daily_stats:
            print("无统计数据")
            return
            
        # 转换为DataFrame
        df_stats = pd.DataFrame(daily_stats)
        
        # 保存文件
        filename = f'T形态汇总统计_{start_date}_{end_date}.csv'
        df_stats.to_csv(filename, index=False, encoding='utf-8-sig')
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
            
        # 生成文件名
        filename = f'T形态详细数据_{latest_date}.csv'
        
        # 保存文件
        latest_data.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"已生成详细文件: {filename}")
        
    except Exception as e:
        print(f"生成详细数据错误: {e}")

def init(context):
    """策略初始化函数"""
    try:
        # 设置分析参数
        context.T_PATTERN_THRESHOLD_L1 = 0.15
        context.T_PATTERN_THRESHOLD_L2 = 0.20
        context.MORNING_T_THRESHOLD_L1 = 0.08
        context.MORNING_T_THRESHOLD_L2 = 0.10
        context.CONTINUOUS_WINDOWS_L1 = 3
        context.CONTINUOUS_WINDOWS_L2 = 6

        # 订阅1分钟K线数据
        subscribe(
            symbols=['SZSE.300353','SHSE.600101','SHSE.600622'],
            frequency='60s',
            count=1,
            unsubscribe_previous=True
        )
        
        # 执行分析
        for symbol in context.symbols:
            analyze_t_patterns(context, symbol)
            
    except Exception as e:
        print(f"初始化错误: {e}")

if __name__ == '__main__':
    run(strategy_id='strategy_id',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='{{token}}',
        backtest_start_time='2024-06-01 09:30:00',
        backtest_end_time='2024-11-13 15:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000)

