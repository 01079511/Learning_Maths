# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import xgboost as xgb
import talib
from sklearn.calibration import CalibratedClassifierCV

# 全局参数
TRAIN_WINDOW = 180  # 训练窗口(6个月)
PREDICT_WINDOW = 15  # 预测窗口(15天)
ROLLING_STEP = 15  # 滚动步长
MIN_SAMPLES = 120  # 最小样本量要求

# 测试股票池
TEST_STOCKS = ['SHSE.600000', 'SHSE.601318']  # 浦发银行和中国平安


def calculate_features(stock_data):
    """计算技术指标特征"""
    if len(stock_data) < MIN_SAMPLES:
        print(f"数据样本量不足: {len(stock_data)} < {MIN_SAMPLES}")
        return None

    try:
        df = pd.DataFrame()
        # 转换数据类型为float
        close = stock_data['close'].astype(float).values
        high = stock_data['high'].astype(float).values
        low = stock_data['low'].astype(float).values
        volume = stock_data['volume'].astype(float).values

        # 趋势指标
        df['ma5'] = talib.MA(close, timeperiod=5)
        df['ma10'] = talib.MA(close, timeperiod=10)
        df['ma20'] = talib.MA(close, timeperiod=20)

        # 动量指标
        df['rsi'] = talib.RSI(close)
        macd, signal, hist = talib.MACD(close)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist

        # KDJ指标
        k, d = talib.STOCH(high, low, close)
        df['kdj_k'] = k
        df['kdj_d'] = d
        df['kdj_j'] = 3 * k - 2 * d

        # 波动率指标
        df['atr'] = talib.ATR(high, low, close)
        df['volatility'] = pd.Series(close).pct_change().rolling(window=20).std()

        # 成交量指标
        df['volume_ma5'] = talib.MA(volume, timeperiod=5)
        df['volume_ma20'] = talib.MA(volume, timeperiod=20)

        df.index = stock_data.index
        df = df.dropna()

        print(f"特征计算完成, shape={df.shape}")
        return df

    except Exception as e:
        print(f"特征计算错误: {e}")
        return None


def get_market_features():
    """获取市场特征数据"""
    try:
        index_data = history_n(symbol='SHSE.000001', frequency='1d',
                               count=TRAIN_WINDOW, fields='close,volume', df=True)

        if index_data is None or len(index_data) < MIN_SAMPLES:
            print("市场数据获取失败或样本量不足")
            return None

        features = pd.DataFrame()
        features['market_return'] = index_data['close'].pct_change()
        features['market_vol'] = features['market_return'].rolling(20).std()
        features['market_volume'] = index_data['volume'].pct_change()

        features.index = index_data.index
        features = features.dropna()

        print(f"市场特征计算完成, shape={features.shape}")
        return features

    except Exception as e:
        print(f"市场特征计算错误: {e}")
        return None


def train_model(stock_data, market_features=None):
    """训练模型"""
    try:
        # 计算个股特征
        stock_features = calculate_features(stock_data)
        if stock_features is None:
            return None, None

        # 生成标签
        future_returns = stock_data['close'].pct_change(PREDICT_WINDOW).shift(-PREDICT_WINDOW)
        labels = (future_returns > 0).astype(int)

        # 对齐数据
        common_index = stock_features.index.intersection(labels.index[:-PREDICT_WINDOW])
        stock_features = stock_features.loc[common_index]
        labels = labels.loc[common_index]

        # 合并市场特征
        if market_features is not None:
            market_features = market_features.reindex(common_index)
            features = pd.concat([stock_features, market_features], axis=1)
        else:
            features = stock_features

        # 删除含有NaN的行
        valid_mask = features.notna().all(axis=1)
        features = features[valid_mask]
        labels = labels[valid_mask]

        if len(features) < MIN_SAMPLES:
            print(f"有效样本数{len(features)}小于最小要求{MIN_SAMPLES}")
            return None, None

        # 特征标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        y = labels.values

        # XGBoost模型
        base_model = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            scale_pos_weight=1
        )

        # 概率校准
        model = CalibratedClassifierCV(base_model, cv=5)
        model.fit(X, y)

        print(f"模型训练完成: 特征shape={X.shape}, 标签shape={y.shape}")
        return model, scaler

    except Exception as e:
        print(f"模型训练错误: {e}")
        return None, None


def evaluate_model(context, stock):
    """评估单个股票模型"""
    try:
        if stock not in context.models:
            print(f"{stock}模型不存在")
            return

        test_data = history_n(symbol=stock, frequency='1d', count=30,
                              fields='symbol,eob,open,close,high,low,volume', df=True)

        features = calculate_features(test_data)
        if features is None:
            return

        X = context.scalers[stock].transform(features)
        probs = context.models[stock].predict_proba(X)

        print(f"\n{stock} 预测结果:")
        print("最近5天上涨概率:", probs[-5:, 1])

        # 计算实际涨跌
        returns = test_data['close'].pct_change()
        print("实际涨跌:", returns[-6:-1].values)

    except Exception as e:
        print(f"模型评估错误: {e}")


def init(context):
    """初始化策略"""
    print("开始初始化策略...")

    # 设置测试股票池
    context.stock_pool = TEST_STOCKS
    context.models = {}
    context.scalers = {}

    # 获取市场特征
    context.market_features = get_market_features()

    # 训练测试股票的模型
    for stock in context.stock_pool:
        print(f"\n处理股票: {stock}")
        stock_data = history_n(symbol=stock, frequency='1d', count=TRAIN_WINDOW,
                               fields='symbol,eob,open,close,high,low,volume', df=True)

        if stock_data is None:
            print(f"{stock} 无数据")
            continue

        print(f"获取数据 shape={stock_data.shape}")

        model, scaler = train_model(stock_data, context.market_features)
        if model is not None:
            context.models[stock] = model
            context.scalers[stock] = scaler
            print(f"{stock} 模型训练成功")
            evaluate_model(context, stock)
        else:
            print(f"{stock} 模型训练失败")

    print("初始化完成")


if __name__ == '__main__':
    run(strategy_id='strategy_id',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='{{token}}',
        backtest_start_time='2023-01-23 08:00:00',
        backtest_end_time='2024-01-23 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000)