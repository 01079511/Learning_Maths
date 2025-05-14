import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime

# 模拟计算截至2025年5月9日的40日收益差 - 尝试中证红利指数
print("=== 尝试使用中证红利指数与万得全A指数计算40日收益差 ===")

# 设置日期范围
start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 5, 9)

start_date_str = start_date.strftime('%Y%m%d')
end_date_str = end_date.strftime('%Y%m%d')

print(f"计算从 {start_date_str} 到 {end_date_str} 的数据")

# 尝试常见的红利指数代码
red_index_codes = [
    "000922",  # 中证红利指数
    "H30533",  # 中证红利全收益指数 (可能需要调整)
    "000015",  # 红利低波
    "399324"   # 深证红利
]

# 万得全A指数代码
WIND_ALL_A_CODE = "881001"

# 测试多个指数代码
for code in red_index_codes:
    print(f"\n---------- 测试指数代码: {code} ----------")
    try:
        # 获取红利相关指数数据
        print(f"获取指数数据 (代码: {code})...")
        red_index_df = ak.index_zh_a_hist(symbol=code, period="daily", 
                                         start_date=start_date_str, end_date=end_date_str)
        
        if red_index_df is None or red_index_df.empty:
            print(f"指数代码 {code} 无法获取到数据，跳过...")
            continue
            
        # 获取万得全A指数数据
        print(f"获取万得全A指数数据 (代码: {WIND_ALL_A_CODE})...")
        wind_all_a_df = ak.index_zh_a_hist(symbol=WIND_ALL_A_CODE, period="daily", 
                                          start_date=start_date_str, end_date=end_date_str)
        
        # 显示获取到的数据信息
        print(f"指数 {code}: 获取到 {len(red_index_df)} 行数据，日期范围: {red_index_df['日期'].min()} 到 {red_index_df['日期'].max()}")
        print(f"万得全A指数: 获取到 {len(wind_all_a_df)} 行数据，日期范围: {wind_all_a_df['日期'].min()} 到 {wind_all_a_df['日期'].max()}")
        
        # 标准化列名
        if '日期' in red_index_df.columns and '收盘' in red_index_df.columns:
            red_index_df = red_index_df.rename(columns={'日期': 'date', '收盘': 'close'})
        
        if '日期' in wind_all_a_df.columns and '收盘' in wind_all_a_df.columns:
            wind_all_a_df = wind_all_a_df.rename(columns={'日期': 'date', '收盘': 'close'})
        
        # 确保日期格式一致
        red_index_df['date'] = pd.to_datetime(red_index_df['date'])
        wind_all_a_df['date'] = pd.to_datetime(wind_all_a_df['date'])
        
        # 设置日期为索引
        red_index_df.set_index('date', inplace=True)
        wind_all_a_df.set_index('date', inplace=True)
        
        # 确保两个数据框有相同的日期范围
        common_dates = red_index_df.index.intersection(wind_all_a_df.index)
        red_index_df = red_index_df.loc[common_dates]
        wind_all_a_df = wind_all_a_df.loc[common_dates]
        
        print(f"共有 {len(common_dates)} 个交易日的数据")
        
        # 尝试两种不同的40日收益计算方法
        
        # 方法1: 几何平均(之前使用的方法)
        print("计算方法1: 使用几何平均计算40日收益...")
        red_index_df['daily_return'] = red_index_df['close'].pct_change()
        wind_all_a_df['daily_return'] = wind_all_a_df['close'].pct_change()
        
        red_index_df['40d_return_1'] = (1 + red_index_df['daily_return']).rolling(window=40).apply(lambda x: np.prod(x) - 1)
        wind_all_a_df['40d_return_1'] = (1 + wind_all_a_df['daily_return']).rolling(window=40).apply(lambda x: np.prod(x) - 1)
        
        # 方法2: 直接计算当前价格与40天前价格的百分比变化
        print("计算方法2: 使用价格变化百分比计算40日收益...")
        red_index_df['40d_return_2'] = red_index_df['close'].pct_change(periods=40)
        wind_all_a_df['40d_return_2'] = wind_all_a_df['close'].pct_change(periods=40)
        
        # 合并数据
        merged_df = pd.DataFrame({
            f'{code}指数': red_index_df['close'],
            '万得全A指数': wind_all_a_df['close'],
            f'{code}指数_方法1_40日收益率(%)': red_index_df['40d_return_1'] * 100,
            '万得全A指数_方法1_40日收益率(%)': wind_all_a_df['40d_return_1'] * 100,
            f'{code}指数_方法2_40日收益率(%)': red_index_df['40d_return_2'] * 100,
            '万得全A指数_方法2_40日收益率(%)': wind_all_a_df['40d_return_2'] * 100
        })
        
        # 计算收益差
        merged_df['方法1_40日收益差(%)'] = (
            merged_df[f'{code}指数_方法1_40日收益率(%)'] - merged_df['万得全A指数_方法1_40日收益率(%)']
        )
        merged_df['方法2_40日收益差(%)'] = (
            merged_df[f'{code}指数_方法2_40日收益率(%)'] - merged_df['万得全A指数_方法2_40日收益率(%)']
        )
        
        # 查看最后一天的数据（应该是2025年5月9日）
        last_date = merged_df.index.max()
        last_row = merged_df.loc[last_date]
        
        print("\n===== 结果验证 =====")
        print(f"最后一个交易日: {last_date.strftime('%Y-%m-%d')}")
        print(f"指数 {code} 收盘价: {last_row[f'{code}指数']:.2f}")
        print(f"万得全A指数收盘价: {last_row['万得全A指数']:.2f}")
        
        print("\n方法1结果 (几何平均):")
        print(f"{code}指数40日收益率: {last_row[f'{code}指数_方法1_40日收益率(%)']:.2f}%")
        print(f"万得全A指数40日收益率: {last_row['万得全A指数_方法1_40日收益率(%)']:.2f}%")
        print(f"40日收益差: {last_row['方法1_40日收益差(%)']:.2f}%")
        
        print("\n方法2结果 (价格变化百分比):")
        print(f"{code}指数40日收益率: {last_row[f'{code}指数_方法2_40日收益率(%)']:.2f}%")
        print(f"万得全A指数40日收益率: {last_row['万得全A指数_方法2_40日收益率(%)']:.2f}%")
        print(f"40日收益差: {last_row['方法2_40日收益差(%)']:.2f}%")
        
        # 与参考图片中的数值对比
        expected_diff = 5.22
        method1_diff = last_row['方法1_40日收益差(%)']
        method2_diff = last_row['方法2_40日收益差(%)']
        
        print(f"\n预期收益差: {expected_diff}%")
        print(f"方法1结果: {method1_diff:.2f}% (差异: {method1_diff - expected_diff:.2f}%)")
        print(f"方法2结果: {method2_diff:.2f}% (差异: {method2_diff - expected_diff:.2f}%)")
        
        if abs(method1_diff - expected_diff) < 0.5 or abs(method2_diff - expected_diff) < 0.5:
            print("验证成功！找到接近的计算结果")
        else:
            print("验证失败：计算结果仍与预期值有较大差异")
    
    except Exception as e:
        print(f"处理 {code} 数据时出错: {e}")

print("\n测试完成。如果没有找到匹配的结果，可能需要查找其他指数代码或计算方法。")