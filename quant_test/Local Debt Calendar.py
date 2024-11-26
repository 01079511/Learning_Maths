import akshare as ak
import pandas as pd
from datetime import datetime

def export_df_to_csv(df, df_name):
    """将 DataFrame 导出为 CSV 文件"""
    # 获取当前日期，格式为 YYYYMMDD
    today_str = datetime.now().strftime('%Y%m%d')
    # 生成文件名
    filename = f"{df_name}{today_str}.csv"
    # 导出为 CSV 文件
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"已导出文件：{filename}")

# 获取地方政府债券发行日历数据
bond_local_government_issue_cninfo_df = ak.bond_local_government_issue_cninfo(start_date="20241120", end_date="20241126")
# 打印结果
# print(bond_local_government_issue_cninfo_df)

# 导出数据
export_df_to_csv(
    bond_local_government_issue_cninfo_df, 
'bond_local_government_issue_cninfo_df'
)