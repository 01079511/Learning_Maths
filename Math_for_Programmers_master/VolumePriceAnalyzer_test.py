import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class MarketAnalyzer:
    def get_realtime_data(self, stock_code):
        """
        获取股票实时数据
        输入: stock_code: str 股票代码(如: '000001')
        输出: 实时行情数据
        """
        try:
            # 获取全部A股实时行情
            df = ak.stock_zh_a_spot_em()
            # 找到指定股票的数据
            stock_data = df[df['代码'] == stock_code].iloc[0]

            # 转换为统一格式的字典
            return {
                'code': stock_data['代码'],
                'name': stock_data['名称'],
                'close': stock_data['最新价'],
                'vol': stock_data['成交量'],  # 单位:手
                'amount': stock_data['成交额'],  # 单位:元
                'turnover': stock_data['换手率'],  # 单位:%
                'amplitude': stock_data['振幅'],  # 单位:%
                'high': stock_data['最高'],
                'low': stock_data['最低'],
                'open': stock_data['今开'],
                'pre_close': stock_data['昨收'],
                'vol_ratio': stock_data['量比'],
                'pe': stock_data['市盈率-动态'],
                'pb': stock_data['市净率'],
                'total_value': stock_data['总市值'],  # 单位:元
                'circ_value': stock_data['流通市值'],  # 单位:元
            }
        except Exception as e:
            print(f"获取实时数据错误: {e}")
            return None

    def get_hist_data(self, stock_code, start_date=None, end_date=None):
        """
        获取股票历史数据
        输入:
            stock_code: str 股票代码(如: '000001')
            start_date: str 开始日期(如: '20240101')
            end_date: str 结束日期(如: '20240109')
        输出:
            DataFrame 历史行情数据
        """
        try:
            # 获取股票日K数据（默认前复权）
            df = ak.stock_zh_a_hist(symbol=stock_code,
                                    period="daily",
                                    start_date=start_date,
                                    end_date=end_date,
                                    adjust="qfq")  # 使用前复权数据

            if df.empty:
                print(f"获取到的数据为空: {stock_code}")
                return None

            # 确保日期格式统一
            df['日期'] = pd.to_datetime(df['日期'])

            return df

        except Exception as e:
            print(f"获取历史数据错误: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return None

    def get_latest_trading_date(self):
        """获取最新交易日期"""
        try:
            # 获取任意股票的最新数据
            df = ak.stock_zh_a_hist(symbol='000001',
                                    period='daily',
                                    end_date=datetime.now().strftime('%Y%m%d'),
                                    adjust='qfq')
            if not df.empty:
                return df['日期'].max()
        except:
            pass
        return None


class VolumePriceScorer(MarketAnalyzer):
    def calculate_score(self, stock_code, target_date=None):
        """
        计算量价配合度评分
        参数:
            stock_code: str, 股票代码 如'000001'
            target_date: str, 可选，指定日期如'20240109'，None则使用最新数据
        返回:
            dict: 评分详情
        """
        score_details = {
            'stock_code': stock_code,
            'trade_date': target_date or datetime.now().strftime('%Y%m%d'),
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
            if target_date is None:
                # 使用实时数据
                data = self.get_realtime_data(stock_code)
                if data is None:
                    return score_details

                # 计算实时量价关系
                price_change = (data['close'] - data['pre_close']) / data['pre_close'] * 100  # 转为百分比
                vol_ratio = data['vol_ratio']

            else:
                # 使用历史数据，获取前20个交易日数据
                end_date = pd.to_datetime(target_date)
                start_date = (end_date - pd.Timedelta(days=30)).strftime('%Y%m%d')
                df = self.get_hist_data(stock_code, start_date, target_date)

                if df is None or len(df) < 2:
                    return score_details

                # 计算历史量价关系
                price_change = df['涨跌幅'].iloc[-1]  # 已经是百分比
                vol_ratio = df['成交量'].iloc[-1] / df['成交量'].iloc[-6:-1].mean()

            # 1. 计算量价关系得分 (40分)
            if price_change > 0 and vol_ratio > 1:
                score_details['score_details']['量价关系得分']['得分'] = 40
                score_details['score_details']['量价关系得分']['描述'] = '价升量增，趋势强劲'
            elif price_change < 0 and vol_ratio < 1:
                score_details['score_details']['量价关系得分']['得分'] = 30
                score_details['score_details']['量价关系得分']['描述'] = '价跌量缩，防御较好'
            else:
                score_details['score_details']['量价关系得分']['得分'] = 20
                score_details['score_details']['量价关系得分']['描述'] = '价量分歧，需要观察'

            # 2. 计算量能水平得分 (30分)
            volume_score = min(vol_ratio * 10, 30)
            score_details['score_details']['量能水平得分']['得分'] = volume_score
            score_details['score_details']['量能水平得分']['描述'] = f'量比 {vol_ratio:.2f}'

            # 3. 计算持续性得分 (20分)
            if target_date is None:
                # 实时数据中使用60日涨跌幅来评估持续性
                hist_df = self.get_hist_data(stock_code,
                                             (datetime.now() - timedelta(days=70)).strftime('%Y%m%d'),
                                             datetime.now().strftime('%Y%m%d'))
                if hist_df is not None:
                    continuous_days = self.calculate_continuous_days(hist_df)
                else:
                    continuous_days = 1
            else:
                continuous_days = self.calculate_continuous_days(df)

            continuity_score = min(continuous_days * 5, 20)
            score_details['score_details']['持续性得分']['得分'] = continuity_score
            score_details['score_details']['持续性得分']['描述'] = f'趋势持续{continuous_days}天'

            # 4. 计算市场环境得分 (10分)
            market_score = self.calculate_market_score(target_date)
            score_details['score_details']['市场环境得分']['得分'] = market_score
            score_details['score_details']['市场环境得分']['描述'] = \
                f'市场环境{"向好" if market_score > 5 else "一般"}'

            # 计算总分
            total_score = sum(item['得分'] for item in score_details['score_details'].values())
            score_details['总分'] = total_score

            # 生成建议
            self.generate_advice(score_details)

            return score_details

        except Exception as e:
            print(f"计算分数错误: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return score_details

    def calculate_continuous_days(self, df):
        """计算价量同向持续天数"""
        continuous_days = 1
        for i in range(len(df) - 2, -1, -1):
            price_up = df['涨跌幅'].iloc[i + 1] > 0
            volume_up = df['成交量'].iloc[i + 1] > df['成交量'].iloc[i]
            if price_up == volume_up:
                continuous_days += 1
            else:
                break
        return continuous_days

    def calculate_market_score(self, target_date=None):
        """计算市场环境得分"""
        try:
            if target_date is None:
                # 获取实时指数数据
                market_data = self.get_realtime_data('000001')  # 上证指数
                if market_data is None:
                    return 5
                market_score = 5 if market_data['close'] > market_data['pre_close'] else 0
                market_score += 5 if market_data['vol_ratio'] > 1 else 0
            else:
                # 获取历史指数数据
                end_date = pd.to_datetime(target_date)
                start_date = (end_date - pd.Timedelta(days=10)).strftime('%Y%m%d')
                market_df = self.get_hist_data('000001', start_date, target_date)

                if market_df is None or len(market_df) < 2:
                    return 5

                market_score = 5 if market_df['涨跌幅'].iloc[-1] > 0 else 0
                market_score += 5 if market_df['成交量'].iloc[-1] > market_df['成交量'].iloc[-2] else 0

            return market_score

        except Exception as e:
            print(f"计算市场环境分数错误: {e}")
            return 5

    def generate_advice(self, score_details):
        """生成建议"""
        total_score = score_details['总分']

        if total_score >= 80:
            score_details['建议'] = '强势股票，可以重点关注'
            score_details['操作建议'] = {
                '建仓仓位': '60-80%',
                '止损位': '3-5%',
                '持仓时间': '5-7天'
            }
            score_details['风险提示'] = '注意及时止盈，严格执行止损'
        elif total_score >= 60:
            score_details['建议'] = '较强势，可以适量参与'
            score_details['操作建议'] = {
                '建仓仓位': '30-50%',
                '止损位': '2-3%',
                '持仓时间': '3-5天'
            }
            score_details['风险提示'] = '注意控制仓位，设置止损'
        else:
            score_details['建议'] = '偏弱势，建议观望'
            score_details['操作建议'] = {
                '建仓仓位': '0-20%',
                '止损位': '2%',
                '持仓时间': '1-2天'
            }
            score_details['风险提示'] = '市场走势不明确，建议谨慎参与'


if __name__ == "__main__":
    # 评分系统初始化
    scorer = VolumePriceScorer()

    # 直接提供股票代码和日期
    stock_code = '300353'  # 股票代码
    target_date = '20241108'  # 分析日期

    # 获取评分
    score = scorer.calculate_score(stock_code, target_date)

    if score:
        print(f"\n股票: {stock_code} 日期: {target_date}")
        print(f"总分: {score['总分']}")
        print("\n详细得分:")
        for name, details in score['score_details'].items():
            print(f"{name}: {details['得分']}/{details['满分']} - {details['描述']}")
        print(f"\n建议: {score['建议']}")
        print(f"操作建议: {score['操作建议']}")
        print(f"风险提示: {score['风险提示']}")
    else:
        print(f"获取 {stock_code} 在 {target_date} 的评分数据失败")