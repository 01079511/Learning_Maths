import akshare as ak
import pandas as pd

# 设置显示选项
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 设置显示宽度为无限制
pd.set_option('display.max_colwidth', None)  # 设置列宽为无限制

# 获取数据
stock_zh_index_spot_em_df = ak.stock_zh_index_spot_em(symbol="上证系列指数")
print(stock_zh_index_spot_em_df)