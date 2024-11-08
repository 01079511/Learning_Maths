import tushare as ts
import pandas as pd
import numpy as np


class MarketAnalyzer:
    # 简化为仅沪深主板指数
    MARKET_INDEX_MAP = {
        'SSE': '000001.SH',  # 上证指数
        'SZSE': '399001.SZ'  # 深证成指
    }

    def __init__(self, token):
        ts.set_token(token)
        self.pro = ts.pro_api()
        # 获取股票基础信息列表
        self.stock_info = self.get_stock_base_info()

    def get_stock_base_info(self):
        """获取主板股票基础信息"""
        try:
            # 获取所有上市状态的主板股票
            df = self.pro.stock_basic(
                exchange='',  # 为空则包含所有交易所
                list_status='L',  # 上市状态：L上市
                fields='ts_code,symbol,name,market,exchange'
            )
            # 仅保留主板股票
            return df[df['market'] == '主板']
        except Exception as e:
            print(f"获取股票基础信息错误: {e}")
            return pd.DataFrame()

    def get_market_type(self, ts_code):
        """
        判断股票所属市场

        参数:
            ts_code: str, tushare股票代码(例如: '600000.SH')
        返回:
            str: 'SSE'(上交所) 或 'SZSE'(深交所) 或 None
        """
        if not self.stock_info.empty:
            stock = self.stock_info[self.stock_info['ts_code'] == ts_code]
            if not stock.empty:
                return stock.iloc[0]['exchange']
        return None

    def get_stock_data(self, ts_code, start_date, end_date):
        """获取股票日线数据"""
        try:
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,open,high,low,close,pre_close,vol,amount'
            )
            return df.sort_values('trade_date')
        except Exception as e:
            print(f"获取股票数据错误: {e}")
            return None


class VolumePriceScorer(MarketAnalyzer):
    def volume_price_score(self, ts_code, target_date, lookback_days=20):
        """计算量价配合度评分"""
        # 验证是否为主板股票
        market_type = self.get_market_type(ts_code)
        if market_type not in ['SSE', 'SZSE']:
            print(f"股票 {ts_code} 不是沪深主板股票")
            return None

        score_details = {
            'ts_code': ts_code,
            'market': market_type,
            'trade_date': target_date,
            'score_details': {
                '量价关系得分': {
                    '满分': 40,
                    '得分': 0,
                    '描述': ''
                },
                '量能水平得分': {
                    '满分': 30,
                    '得分': 0,
                    '描述': ''
                },
                '持续性得分': {
                    '满分': 20,
                    '得分': 0,
                    '描述': ''
                },
                '市场环境得分': {
                    '满分': 10,
                    '得分': 0,
                    '描述': ''
                }
            },
            '总分': 0,
            '建议': '',
            '风险提示': '',
            '操作建议': {
                '建仓仓位': '',
                '止损位': '',
                '持仓时间': ''
            }
        }

        try:
            # 获取数据
            start_date = (pd.to_datetime(target_date) -
                          pd.Timedelta(days=lookback_days + 10)).strftime('%Y%m%d')

            # 获取股票数据
            df = self.get_stock_data(ts_code, start_date, target_date)
            if df is None or len(df) < lookback_days:
                return score_details

            # 计算量价关系得分
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            volume_change = (df['vol'].iloc[-1] - df['vol'].iloc[-2]) / df['vol'].iloc[-2]

            if price_change > 0 and volume_change > 0:
                score_details['score_details']['量价关系得分']['得分'] = 40
                score_details['score_details']['量价关系得分']['描述'] = '价升量增，趋势强劲'
            elif price_change < 0 and volume_change < 0:
                score_details['score_details']['量价关系得分']['得分'] = 30
                score_details['score_details']['量价关系得分']['描述'] = '价跌量缩，防御较好'
            else:
                score_details['score_details']['量价关系得分']['得分'] = 20
                score_details['score_details']['量价关系得分']['描述'] = '价量分歧，需要观察'

            # 计算量能水平得分
            vol_ma5 = df['vol'].rolling(5).mean().iloc[-1]
            vol_ratio = df['vol'].iloc[-1] / vol_ma5
            volume_score = min(vol_ratio * 10, 30)
            score_details['score_details']['量能水平得分']['得分'] = volume_score
            score_details['score_details']['量能水平得分']['描述'] = f'量能是5日均量的{vol_ratio:.2f}倍'

            # ... [其余评分逻辑保持不变] ...

            return score_details

        except Exception as e:
            print(f"计算分数错误: {e}")
            return score_details


# 使用示例
if __name__ == "__main__":
    scorer = VolumePriceScorer('54bb7c652524b9994d05617b76f2abb336f737f10b3be680a6fafe85')

    test_stocks = [
        '000702.SZ',  # 深证主板
        '002449.SZ',  # 深证主板
    ]

    results = []
    for stock in test_stocks:
        score = scorer.volume_price_score(stock, '20241108')
        if score:  # 只处理有效的主板股票
            results.append(scorer.format_output(score))

    if results:
        final_result = pd.concat(results)
        print(final_result)