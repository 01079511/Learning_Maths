# 用于数据处理
import numpy as np
import pandas as pd
# 用于获取数据
import akshare as ak
# 导入线性回归模型
# from sklearn.linear_model import LinearRegression
from tabulate import tabulate
# 设置忽略警告
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta

# 设置查询日期
current_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
start_date = "20150101"

# 获取短融ETF数据
# fund_data = ak.fund_etf_fund_daily_em() 实时净值
fund_his_data = ak.fund_etf_fund_info_em(fund="511360", start_date=start_date, end_date=current_date)

# 只需要净值日期 单位净值 累计净值
DF = fund_his_data[['单位净值', '累计净值','日增长率']].rename(
    columns={
        '单位净值': 'nav',  
        '累计净值': 'acc_nav',
        '日增长率': 'dir'  # daily increase rate -> dir
    }
)

# 将Index设置为datetime格式的日期
DF.index = pd.to_datetime(fund_his_data['净值日期']).tolist()

# 去除空值
DF = DF.dropna()

# 打印数据检查清洗结果
# print(fund_his_data.tail())
# print(DF.tail())

def prepare_data(DF):
    """
    预处理：计算日增长率 dir、填充缺失值
    """
    DF['dir'] = DF['nav'].pct_change()  # (今日nav - 昨日nav) / 昨日nav
    DF.dropna(inplace=True)
    return DF

def backtest_growth_prediction(DF, window=6):
    """
    基于 recent_window 天的平均增长率(可使用EWMA或简单滚动平均)做回测
    """
    DF = prepare_data(DF.copy())

    # 新增一列，用来存放对下一日 nav 的预测
    DF['nav_pred'] = np.nan

    # 逐日回测
    for i in range(window, len(DF)):
        # 取前 window 天的日增长率
        recent_dir = DF['dir'].iloc[i-window:i]

        # 方式1：简单滚动平均
        mean_dir = recent_dir.mean()

        # 方式2：EWMA
        # mean_dir = recent_dir.ewm(span=window).mean().iloc[-1]

        # 明日预测净值 = 今日净值 * (1 + mean_dir)
        # 由于当前行 i 代表今日，因此要对下一日做预测
        DF['nav_pred'].iloc[i] = DF['nav'].iloc[i-1] * (1 + mean_dir)

    # 计算误差: 当日实际 nav 与预测 nav 的差距
    DF['error'] = (DF['nav'] - DF['nav_pred']).abs()
    DF.dropna(inplace=True)
    
    # 评估策略，如平均误差、MAPE等
    mae = DF['error'].mean()
    mape = (DF['error'] / DF['nav']).mean() * 100

    print(f"回测区间共 {len(DF)} 条记录")
    print(f"平均绝对误差 MAE: {mae:.6f}")
    print(f"平均绝对百分比误差 MAPE: {mape:.2f}%")

    return DF[['nav','nav_pred','dir','error']]

backtested_df = backtest_growth_prediction(DF, window=6)
print(backtested_df.tail())