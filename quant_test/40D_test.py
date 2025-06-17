import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib.ticker as mtick
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import platform

# 解决中文显示问题
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置日期范围
end_date = datetime.now()
start_date = end_date - timedelta(days=1825)  # 约5年数据
start_date_str = start_date.strftime('%Y%m%d')
end_date_str = end_date.strftime('%Y%m%d')

# 指数代码（必须使用全收益指数）
RED_LOW_VOL_CODE = "000922"  # 中证红利全收益指数（示例代码，需根据AKShare调整）
WIND_ALL_A_CODE = "881001"   # 万得全A指数（示例代码）

# 数据获取函数（需确认AKShare接口支持全收益指数）
def get_index_data(index_name, index_code):
    """获取指数日线数据（收盘价）"""
    try:
        df = ak.index_zh_a_hist(symbol=index_code, period="daily", 
                               start_date=start_date_str, end_date=end_date_str)
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        df.rename(columns={'收盘': index_name}, inplace=True)
        return df[[index_name]]
    except Exception as e:
        print(f"获取 {index_name} 数据失败: {str(e)}")
        return None

# 主程序
try:
    # 获取数据
    red_low_vol_df = get_index_data("中证红利全收益", RED_LOW_VOL_CODE)
    wind_all_a_df = get_index_data("万得全A", WIND_ALL_A_CODE)
    
    # 数据对齐处理
    merged_df = pd.concat([red_low_vol_df, wind_all_a_df], axis=1)
    merged_df = merged_df.resample('D').ffill().dropna()  # 按日填充对齐
    
    # 计算40日简单涨跌幅差（原文逻辑）
    merged_df['红利40日收益'] = merged_df['中证红利全收益'].pct_change(periods=40) * 100
    merged_df['全A40日收益'] = merged_df['万得全A'].pct_change(periods=40) * 100
    merged_df['40日收益差(%)'] = merged_df['红利40日收益'] - merged_df['全A40日收益']
    
    # 计算252日均线
    merged_df['收益差MA252(%)'] = merged_df['40日收益差(%)'].rolling(252).mean()
    
    # 提取最新收益差
    latest_return_diff = merged_df['40日收益差(%)'].iloc[-1]
    
    # 输出关键数据（修复格式化输出）
    print(f"\n最新40日收益差: {latest_return_diff:.2f}%")  # 格式化保留两位小数
    print(f"数据日期范围: {merged_df.index.min()} 至 {merged_df.index.max()}")
    
    # 可视化配置
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
    fig.suptitle('中证红利全收益 vs 万得全A 40日收益差', fontsize=16, y=0.95)
    
    # 子图1：收益差与均线
    ax1.plot(merged_df.index, merged_df['40日收益差(%)'], 'r-', linewidth=1.5, label='40日收益差')
    ax1.plot(merged_df.index, merged_df['收益差MA252(%)'], 'b-', linewidth=1.5, label='252日均线')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('收益差 (%)')
    
    # 子图2：指数走势对比
    ax2.plot(merged_df.index, merged_df['中证红利全收益'], '#008080', label='中证红利全收益')
    ax2_right = ax2.twinx()
    ax2_right.plot(merged_df.index, merged_df['万得全A'], '#FFA500', label='万得全A')
    ax2.set_ylabel('中证红利指数', color='#008080')
    ax2_right.set_ylabel('万得全A指数', color='#FFA500')
    ax2.grid(True, alpha=0.3)
    
    # 格式化日期轴
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=0)
    
    # 数据来源标注
    plt.figtext(0.5, 0.01, f"数据来源：Wind，截至{datetime.now().strftime('%Y-%m-%d')}", 
                ha='center', fontsize=10)
    
    # 保存结果
    merged_df.to_csv('红利低波vs万得全A_收益差.csv')
    plt.savefig('红利低波vs万得全A_收益差.png', dpi=300, bbox_inches='tight')
    plt.show()

except Exception as e:
    print("程序运行错误:", str(e))