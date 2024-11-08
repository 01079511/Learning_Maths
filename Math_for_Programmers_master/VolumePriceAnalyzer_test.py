import tushare as ts
import pandas as pd
import numpy as np


class VolumePriceAnalyzer:
    def __init__(self, token):
        # 初始化Tushare
        ts.set_token(token)
        self.pro = ts.pro_api()

    def get_stock_data(self, stock_code, start_date, end_date):
        """获取股票数据"""
        try:
            # 获取日线数据
            df = self.pro.daily(ts_code=stock_code,
                                start_date=start_date,
                                end_date=end_date)
            # 按日期正序排列
            df = df.sort_values('trade_date')
            return df
        except Exception as e:
            print(f"获取数据错误: {e}")
            return None

    def volume_price_score(self, stock_code, date, lookback_days=20):
        """
        计算量价配合度评分

        参数:
        stock_code: 股票代码
        date: 计算日期
        lookback_days: 回溯天数
        """
        score = 0

        # 获取历史数据
        start_date = pd.to_datetime(date) - pd.Timedelta(days=lookback_days + 10)
        end_date = date
        df = self.get_stock_data(stock_code, start_date.strftime('%Y%m%d'), end_date)

        if df is None or len(df) < lookback_days:
            return 0

        # 计算基础指标
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['vol'].pct_change()

        # 获取当日数据
        current_day = df.iloc[-1]
        prev_day = df.iloc[-2]

        try:
            # 1. 基础量价关系 (40分)
            price_up = current_day['close'] > prev_day['close']
            volume_up = current_day['vol'] > prev_day['vol']

            if price_up and volume_up:
                score += 40
            elif (not price_up) and (not volume_up):
                score += 30

            # 2. 量能水平 (30分)
            avg_volume = df['vol'].rolling(5).mean().iloc[-1]
            vol_ratio = current_day['vol'] / avg_volume
            score += min(vol_ratio * 10, 30)

            # 3. 持续性 (20分)
            def check_continuous_days():
                continuous = 0
                trend = 'up' if price_up else 'down'

                for i in range(len(df) - 1, -1, -1):
                    day_price_up = df['price_change'].iloc[i] > 0
                    day_volume_up = df['volume_change'].iloc[i] > 0

                    if trend == 'up':
                        if day_price_up and day_volume_up:
                            continuous += 1
                        else:
                            break
                    else:
                        if (not day_price_up) and (not day_volume_up):
                            continuous += 1
                        else:
                            break

                return continuous

            continuous_days = check_continuous_days()
            score += min(continuous_days * 5, 20)

            # 4. 市场环境 (10分)
            # 获取上证指数数据
            index_df = self.pro.index_daily(ts_code='000001.SH',
                                            start_date=start_date.strftime('%Y%m%d'),
                                            end_date=end_date)
            if not index_df.empty:
                index_df = index_df.sort_values('trade_date')
                market_trend_good = index_df['close'].iloc[-1] > index_df['close'].iloc[-2]
                if market_trend_good:
                    score += 10

            return round(score, 2)

        except Exception as e:
            print(f"计算分数错误: {e}")
            return 0

    def batch_analyze(self, stock_list, date):
        """批量分析股票列表"""
        results = []
        for stock_code in stock_list:
            score = self.volume_price_score(stock_code, date)
            results.append({
                'stock_code': stock_code,
                'date': date,
                'score': score
            })
        return pd.DataFrame(results)


# 使用示例
if __name__ == "__main__":
    # 初始化分析器
    analyzer = VolumePriceAnalyzer('54bb7c652524b9994d05617b76f2abb336f737f10b3be680a6fafe85')

    # 单只股票分析
    #score = analyzer.volume_price_score('000001.SZ', '20241109')
    #print(f"量价配合度评分: {score}")

    # 批量分析
    stock_list = ['000702.SZ', '002449.SZ']
    results = analyzer.batch_analyze(stock_list, '20241109')
    print("\n批量分析结果:")
    print(results)
