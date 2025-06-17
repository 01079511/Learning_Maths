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

# 如果系统中没有合适的中文字体，可以使用不依赖特定字体的方式显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 打印可用字体，帮助调试
print("系统中可用的部分字体:")
for font in list(set([f.name for f in fm.fontManager.ttflist]))[:10]:  # 只显示前10个
    print(f"- {font}")

# 设置日期范围 - 获取近5年数据以匹配图片中的时间范围
end_date = datetime.now()
start_date = end_date - timedelta(days=1825)  # 约5年数据
start_date_str = start_date.strftime('%Y%m%d')
end_date_str = end_date.strftime('%Y%m%d')

print(f"获取从 {start_date_str} 到 {end_date_str} 的数据")

# 使用测试确认的代码
RED_LOW_VOL_CODE = "000922"  # 红利低波全收益指数代码
WIND_ALL_A_CODE = "881001"   # 万得全A指数代码

try:
    # 获取红利低波全收益指数数据
    print(f"获取红利低波全收益指数数据 (代码: {RED_LOW_VOL_CODE})...")
    red_low_vol_df = ak.index_zh_a_hist(symbol=RED_LOW_VOL_CODE, period="daily", 
                                       start_date=start_date_str, end_date=end_date_str)
    
    # 获取万得全A指数数据
    print(f"获取万得全A指数数据 (代码: {WIND_ALL_A_CODE})...")
    wind_all_a_df = ak.index_zh_a_hist(symbol=WIND_ALL_A_CODE, period="daily", 
                                      start_date=start_date_str, end_date=end_date_str)
    
    # 标准化列名并打印列名方便调试
    print(f"红利低波指数数据列名: {red_low_vol_df.columns.tolist()}")
    print(f"万得全A指数数据列名: {wind_all_a_df.columns.tolist()}")
    
    # 标准化列名
    if '日期' in red_low_vol_df.columns and '收盘' in red_low_vol_df.columns:
        red_low_vol_df = red_low_vol_df.rename(columns={'日期': 'date', '收盘': 'close'})
    
    if '日期' in wind_all_a_df.columns and '收盘' in wind_all_a_df.columns:
        wind_all_a_df = wind_all_a_df.rename(columns={'日期': 'date', '收盘': 'close'})
    
    # 确保日期格式一致
    red_low_vol_df['date'] = pd.to_datetime(red_low_vol_df['date'])
    wind_all_a_df['date'] = pd.to_datetime(wind_all_a_df['date'])
    
    # 设置日期为索引
    red_low_vol_df.set_index('date', inplace=True)
    wind_all_a_df.set_index('date', inplace=True)
    
    # 确保两个数据框有相同的日期范围
    common_dates = red_low_vol_df.index.intersection(wind_all_a_df.index)
    red_low_vol_df = red_low_vol_df.loc[common_dates]
    wind_all_a_df = wind_all_a_df.loc[common_dates]
    
    print(f"共有 {len(common_dates)} 个交易日的数据")
    
    # 计算每日收益率
    red_low_vol_df['daily_return'] = red_low_vol_df['close'].pct_change()
    wind_all_a_df['daily_return'] = wind_all_a_df['close'].pct_change()
    
    # 计算40日滚动收益率（使用几何收益率而非简单收益率）
    print("计算40日滚动收益...")
    red_low_vol_df['40d_return'] = (1 + red_low_vol_df['daily_return']).rolling(window=40).apply(lambda x: np.prod(x) - 1)
    wind_all_a_df['40d_return'] = (1 + wind_all_a_df['daily_return']).rolling(window=40).apply(lambda x: np.prod(x) - 1)
    
    # 合并数据
    result_df = pd.DataFrame({
        '红利低波指数': red_low_vol_df['close'],
        '万得全A指数': wind_all_a_df['close'],
        '红利低波指数_40日收益率': red_low_vol_df['40d_return'] * 100,  # 转为百分比
        '万得全A指数_40日收益率': wind_all_a_df['40d_return'] * 100,   # 转为百分比
        '40日收益差(%)': (red_low_vol_df['40d_return'] - wind_all_a_df['40d_return']) * 100  # 转为百分比
    })
    
    # 计算收益差的242日移动平均线（约1年交易日）
    result_df['收益差MA242(%)'] = result_df['40日收益差(%)'].rolling(window=242).mean()
    
    # 打印数据范围，便于调试
    print(f"数据日期范围: 从 {result_df.index.min()} 到 {result_df.index.max()}")
    
    # 显示最近的数据
    print("\n最近的收益差数据:")
    print(result_df.tail(10)[['40日收益差(%)', '收益差MA242(%)']].round(2))
    
    # 统计数据的基本指标，以确认数据是否合理
    print("\n收益差基本统计:")
    stats = result_df['40日收益差(%)'].describe().round(2)
    print(stats)
    
    # 计算一些统计数据
    latest_return_diff = result_df['40日收益差(%)'].iloc[-1]
    print(f"\n最新40日收益差: {latest_return_diff:.2f}%")
    print(f"平均收益差: {result_df['40日收益差(%)'].mean():.2f}%")
    print(f"收益差标准差: {result_df['40日收益差(%)'].std():.2f}%")
    print(f"最大收益差: {result_df['40日收益差(%)'].max():.2f}%")
    print(f"最小收益差: {result_df['40日收益差(%)'].min():.2f}%")
    
    # 计算红利低波指数跑赢万得全A指数的频率
    positive_ratio = (result_df['40日收益差(%)'] > 0).mean()
    print(f"红利低波指数跑赢万得全A指数的频率: {positive_ratio:.2%}")
    
    # 导出数据到CSV文件
    result_df.to_csv('红利低波vs万得全A指数收益差数据.csv')
    print("\n数据已导出到 '红利低波vs万得全A指数收益差数据.csv'")
    
    # 创建图表，类似于提供的Excel图片
    print("\n创建图表...")
    
    # 创建包含两个子图的图表，使用更大的高度比
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
    
    # 设置图表标题 - 使用简单文本而非中文，避免字体问题
    title_text = '红利低波全收益相对万得全A40日收益差（单位：%）'
    try:
        fig.suptitle(title_text, fontsize=16, y=0.98)
    except:
        # 如果中文显示有问题，使用英文标题
        fig.suptitle('40-Day Return Difference (Low Vol Dividend vs All A Index)', fontsize=16, y=0.98)
        print("警告: 使用了英文标题，因为中文显示可能有问题")
    
    # 第一个子图：收益差图表
    ax1.plot(result_df.index, result_df['40日收益差(%)'], 'r-', linewidth=1.5)
    ax1.plot(result_df.index, result_df['收益差MA242(%)'], 'blue', linewidth=1.5)
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)  # 添加水平零线
    ax1.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)  # 更浅的网格线
    
    # 使用安全的方式添加Y轴标签
    try:
        ax1.set_ylabel('收益差', fontsize=12)
    except:
        ax1.set_ylabel('Return Diff', fontsize=12)
    
    # 设置y轴刻度为百分比格式并调整范围
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))  # 去掉小数点后的零
    ax1.set_ylim(-15, 15)  # 设置y轴范围为-15%到15%
    
    # 在图表右上角添加最新收益差数值，使用绿色矩形
    latest_return_diff = result_df['40日收益差(%)'].iloc[-1]
    rect = patches.Rectangle((0.87, 0.85), 0.12, 0.1, linewidth=1, edgecolor='green', 
                            facecolor='white', transform=ax1.transAxes, alpha=0.9)
    ax1.add_patch(rect)
    
    # 安全地添加收益差标注
    try:
        ax1.annotate(f'最新收益差：{latest_return_diff:.2f}%',
                   xy=(0.93, 0.9), xycoords='axes fraction',
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=10, color='green')
    except:
        ax1.annotate(f'Latest: {latest_return_diff:.2f}%',
                   xy=(0.93, 0.9), xycoords='axes fraction',
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=10, color='green')
    
    # 添加图例 - 尝试使用中文，如果失败则使用英文
    try:
        ax1.legend(['收益差', 'MA242(收益差)'], loc='upper left', ncol=2, frameon=True, framealpha=0.8, facecolor='white')
    except:
        ax1.legend(['Return Diff', 'MA242'], loc='upper left', ncol=2, frameon=True, framealpha=0.8, facecolor='white')
    
    # 第二个子图：指数点位走势
    # 创建左右双y轴，左侧为红利低波指数，右侧为万得全A指数
    color1, color2 = '#008080', '#FFA500'  # 设置两条线的颜色为蓝绿色和橙黄色
    
    # 左侧y轴 - 红利低波指数
    line1, = ax2.plot(result_df.index, result_df['红利低波指数'], color=color1, linewidth=1.5)
    
    # 安全地添加Y轴标签
    try:
        ax2.set_ylabel('红利低波指数', fontsize=12, color=color1)
    except:
        ax2.set_ylabel('Low Vol Dividend', fontsize=12, color=color1)
    
    ax2.tick_params(axis='y', labelcolor=color1)
    
    # 右侧y轴 - 万得全A指数
    ax3 = ax2.twinx()
    line2, = ax3.plot(result_df.index, result_df['万得全A指数'], color=color2, linewidth=1.5)
    
    # 安全地添加Y轴标签
    try:
        ax3.set_ylabel('万得全A指数', fontsize=12, color=color2)
    except:
        ax3.set_ylabel('All A Index', fontsize=12, color=color2)
    
    ax3.tick_params(axis='y', labelcolor=color2)
    
    # 添加网格线
    ax2.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)  # 更浅的网格线
    
    # 设置x轴格式
    date_format = mdates.DateFormatter('%Y/%m')
    ax2.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每3个月显示一次
    fig.autofmt_xdate(rotation=0, ha='center')  # 不旋转日期标签
    
    # 第二个子图的图例 - 安全方式添加
    try:
        ax2.legend(handles=[line1, line2], labels=['红利低波指数', '万得全A指数'], 
                 loc='upper left', ncol=2, frameon=True, framealpha=0.8, facecolor='white')
    except:
        ax2.legend(handles=[line1, line2], labels=['Low Vol Dividend', 'All A Index'], 
                 loc='upper left', ncol=2, frameon=True, framealpha=0.8, facecolor='white')
    
    # 添加数据来源说明
    try:
        plt.figtext(0.5, 0.01, 
                  f"数据来源：Wind，招商基金整理，截至{datetime.now().strftime('%Y年%m月%d日')}，数据追溯近5年", 
                  ha='center', fontsize=10)
    except:
        plt.figtext(0.5, 0.01, 
                  f"Data Source: Wind, {datetime.now().strftime('%Y-%m-%d')}, 5-year data", 
                  ha='center', fontsize=10)
    
    # 调整子图间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.05)  # 减小子图间距，增大顶部空间
    
    # 保存图表
    plt.savefig('红利低波全收益相对万得全A40日收益差.png', dpi=300, bbox_inches='tight')
    print("已保存图表到 '红利低波全收益相对万得全A40日收益差.png'")
    
    # 显示图表
    plt.show()
    
except Exception as e:
    print(f"处理数据时出错: {e}")