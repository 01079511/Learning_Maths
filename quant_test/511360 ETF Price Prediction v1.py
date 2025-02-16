# 用于数据处理
import numpy as np
import pandas as pd
# 用于获取数据
import akshare as ak
# 导入线性回归模型
from sklearn.linear_model import LinearRegression
# 导入画图库、设置主题和中文显示
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-bright') # 设置绘图主题,参数: seaborn-v0_8 or seaborn-v0_8-bright/ dark/ whitegrid可选    
plt.rcParams['font.sans-serif'] = ['SimHei'] # 中文显示
plt.rcParams['axes.unicode_minus'] = False   # 负数显示
# 设置忽略警告
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta

# 获取短融ETF的历史行情数据
etf_data = ak.fund_etf_hist_em(symbol='511360', period='daily', start_date='20130801', end_date='20241225')
# 只需要收盘价序列
Df = etf_data[['收盘']].rename(columns={'收盘':'Close'})
# 将Index设置为datetime格式的日期
Df.index = pd.to_datetime(etf_data['日期']).tolist()
# 去除空值
Df = Df.dropna()
# 画出短融ETF的价格走势图
Df.Close.plot(figsize=(15, 8), color='red')
plt.ylabel('短融ETF价格')
plt.title('短融ETF价格序列')
# plt.show()

# 计算均线因子
Df['S1'] = Df['Close'].rolling(window=55).mean()
Df['S2'] = Df['Close'].rolling(window=60).mean()
# 第二天的收盘价
Df['next_day_price'] = Df['Close'].shift(-1)
Df = Df.dropna()

# 定义解释变量
X = Df[['S1', 'S2']]
# 定义因变量
y = Df['next_day_price']

# 将数据划分为训练集和测试集
t = int(0.8 * Df.shape[0])

# 训练集
X_train = X.iloc[:t]
y_train = y.iloc[:t]

# 测试集
X_test = X.iloc[t:]
y_test = y.iloc[t:]

# 创建线性回归模型并训练
linear = LinearRegression(fit_intercept=True).fit(X_train, y_train)
print('短融ETF价格(y) = %.2f * 55日移动平均线(x1) \
 %+.2f * 60日移动平均线(x2) \
 %+.2f (constant)' %(linear.coef_[0], linear.coef_[1], linear.intercept_))

 # 预测短融ETF第二日的价格
predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(15, 8))
y_test.plot()
plt.legend(['预测的价格', '实际的价格'])
plt.ylabel('短融ETF的价格')
plt.show()

# 决定系数R2
r2_train = linear.score(X_train, y_train)
r2_test = linear.score(X_test, y_test)
print('训练集决定系数: %.4f' %r2_train)
print('测试集决定系数: %.4f' %r2_test)

# 只考虑测试集当中的数据
cp = pd.DataFrame()
cp['price'] = Df.iloc[t:]['Close']
cp['predicted_price_next_day'] = predicted_price
# 短融ETF的日收益率
cp['cp_returns'] = cp['price'].pct_change()
# 如果预测价格比前一个预测的价格高，则买入，否则卖出或空仓
cp['signal'] = np.where(cp.predicted_price_next_day.shift(1) < cp.predicted_price_next_day, 1, 0)
# 策略的日收益率
cp['strategy_returns'] = cp['signal'].shift(1) * cp['cp_returns']
# 策略和基准的净值曲线
cp['strategy_nv'] = (cp['strategy_returns'] + 1).cumprod()
cp['bmk_nv'] = (cp['cp_returns'] + 1).cumprod()
# 绘制净值曲线图
cp[['strategy_nv','bmk_nv']].plot(figsize=(15, 8), color=['SteelBlue', 'Yellow'],
                                    title='短融ETF价格择时策略净值曲线图 (2013-2024)')
plt.legend(['策略净值', '基准净值'])
plt.ylabel('净值')
plt.show()

# 计算夏普率
strategy_sharpe = cp['strategy_returns'].mean() / cp['strategy_returns'].std() * (252**0.5)
bmk_sharpe = cp['cp_returns'].mean() / cp['cp_returns'].std() * (252**0.5)
print('策略夏普率: %.2f' %strategy_sharpe)
print('基准夏普率: %.2f' %bmk_sharpe)

# 当前日期
# current_date = datetime.now().strftime('%Y%m%d')
current_date = (datetime.now() - timedelta(days=4)).strftime('%Y%m%d') # 获取昨天的日期,测试当日结果

# 获取数据
etf_data = ak.fund_etf_hist_em(symbol='511360', period='daily', start_date='20230101', end_date=current_date)
data = etf_data[['收盘']].rename(columns={'收盘':'Close'})
data.index = pd.to_datetime(etf_data['日期']).tolist()
# 计算均线因子
data['S1'] = data['Close'].rolling(window=55).mean()
data['S2'] = data['Close'].rolling(window=60).mean()
data = data.dropna()
# 预测短融ETF第二天的价格
data['predicted_cp_price'] = linear.predict(data[['S1', 'S2']])
data['signal'] = np.where(data.predicted_cp_price.shift(1) < data.predicted_cp_price, '买入' , '空仓')
# 输出预测值
print(data.tail(1)[['signal','predicted_cp_price']])