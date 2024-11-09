import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

'''
这个评分系统的数理逻辑和价值：

量价关系得分（40分）：

pythonCopyif price_change > 0 and volume_change > 0:
    # 得40分：价升量增
    # 原理：市场买盘积极，需求强于供给
    # 意义：趋势最强，持续性最好

elif price_change < 0 and volume_change < 0:
    # 得30分：价跌量缩
    # 原理：市场供给减少，卖盘意愿不强
    # 意义：防御性好，企稳可能性大

else:  # 价量背离
    # 得20分：价升量减或价跌量增
    # 原理：供需关系不明确，可能是趋势转折信号
    # 意义：需要观察，可能是趋势衰竭

量能水平得分（30分）：

pythonCopy# 计算当前成交量相对5日均量的比值
vol_ratio = current_volume / vol_ma5
volume_score = min(vol_ratio * 10, 30)

# 理论依据：
# vol_ratio = 1.0：正常水平（得10分）
# vol_ratio = 2.0：放量一倍（得20分）
# vol_ratio = 3.0：放量两倍（得30分封顶）

# 实战意义：
- 量比越大表示市场参与度越高
- 但过度放量（>3倍）可能是风险信号
- 持续性放量比单日放量更有价值

持续性得分（20分）：

pythonCopy# 连续同向变动的天数（价量同向）
continuous_days = 计算连续天数
continuity_score = min(continuous_days * 5, 20)

# 评分逻辑：
- 1天：5分
- 2天：10分
- 3天：15分
- 4天及以上：20分（封顶）

# 理论依据：
- 趋势持续时间越长，惯性越强
- 但过长的持续也意味着调整压力增大

市场环境得分（10分）：

pythonCopy# 两个维度各5分：
1. 短期趋势：当日上涨得5分
2. 中期趋势：5日均线>10日均线得5分

# 逻辑：
- 顺应市场大趋势
- 强市场提高个股成功率

综合评判标准：

markdownCopy总分 >= 80：强势股
- 量价配合优秀
- 持续性强
- 市场环境好
- 建议：可以重点关注

总分 60-79：较强势
- 量价基本配合
- 有一定持续性
- 建议：适量参与

总分 < 60：偏弱势
- 量价关系不佳
- 持续性差
- 建议：观望为主
实战价值：

客观量化：

markdownCopy- 将主观判断转化为数据
- 避免情绪化交易
- 便于批量筛选

多维度分析：

markdownCopy- 不仅看涨跌幅
- 综合考虑量价关系
- 考虑市场环境影响

风险控制：

markdownCopy建仓仓位基于得分：
- >=80分：60-80%仓位
- 60-79分：30-50%仓位
- <60分：0-20%仓位

止损设置基于得分：
- >=80分：3-5%止损
- 60-79分：2-3%止损
- <60分：2%止损

实用价值：

markdownCopy1. 选股：
- 批量筛选高分股票
- 优先选择量价配合好的

2. 择时：
- 持续性判断
- 市场环境评估

3. 风控：
- 基于分数的仓位管理
- 动态调整持仓

策略优化方向：

markdownCopy1. 权重调整：
- 可根据市场风格调整各项权重
- 牛市重视量能和持续性
- 熊市重视防御性

2. 阈值优化：
- 根据回测优化各项阈值
- 针对不同市场调整标准
- 动态调整评分标准
局限性：

历史数据不代表未来
需要配合基本面分析
可能存在滞后性
极端行情可能失效

建议：

作为辅助工具使用
结合其他技术指标
考虑基本面因素
注意市场大环境
'''


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
    """
    这个系统支持两种数据获取方式，取决于调用 calculate_score 时是否提供 target_date 参数：

    当提供 target_date 时（如示例中的 '20240109'）：
    使用历史数据（ak.stock_zh_a_hist）
    获取指定日期的历史行情数据进行评分
    适合分析历史某一天的量价配合度

    当不提供 target_date 时（即 target_date=None）：
    使用实时数据（ak.stock_zh_a_spot_em）
    获取当前最新行情数据进行评分
    适合实时市场分析

    在当前主函数（main）的示例中，由于我们明确提供了 target_date='20240109'，
    所以系统使用的是历史数据进行评分。如果想获取实时评分，只需要在调用时不传入 target_date 参数即可。
    """


    # 评分系统初始化
    scorer = VolumePriceScorer()

    # 直接提供股票代码和日期
    stock_code = '300353'  # 股票代码
    target_date = None  # 分析日期,None调用实时数据，过去日期可用于回测

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