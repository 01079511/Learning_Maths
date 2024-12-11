import akshare as ak
import pandas as pd
from datetime import datetime

def get_month_end_stock_data(symbol_name):
    """获取指定指数的月末数据"""
    try:
        # 获取指数的市盈率数据
        stock_index_pe_lg_df = ak.stock_index_pe_lg(symbol=symbol_name)
        
        # 转换日期列为datetime类型
        stock_index_pe_lg_df['日期'] = pd.to_datetime(stock_index_pe_lg_df['日期'])
        
        # 过滤1990年以后的数据
        stock_index_pe_lg_df = stock_index_pe_lg_df[
            stock_index_pe_lg_df['日期'] >= pd.to_datetime('1990-01-01')
        ]
        
        # 按月份分组，获取月末数据
        month_end_df = stock_index_pe_lg_df.loc[
            stock_index_pe_lg_df.groupby(
                stock_index_pe_lg_df['日期'].dt.to_period('M')
            )['日期'].idxmax()
        ].reset_index(drop=True)
        
        return month_end_df
    
    except Exception as e:
        print(f"获取{symbol_name}数据时出错: {str(e)}")
        return pd.DataFrame()

def get_bond_yield_on_dates(dates):
    """根据传入的日期列表过滤国债收益率数据"""
    try:
        # 获取中美国债收益率数据
        bond_zh_us_rate_df = ak.bond_zh_us_rate(start_date="19901219")
        
        # 转换日期列为datetime类型
        bond_zh_us_rate_df['日期'] = pd.to_datetime(bond_zh_us_rate_df['日期'])
        
        # 过滤需要的列，例如 '中国国债收益率(10年)', '美国国债收益率(10年)'
        bond_columns = ['中国国债收益率10年']
        
        # 根据传入的日期列表进行过滤
        filtered_bond_df = bond_zh_us_rate_df[bond_zh_us_rate_df['日期'].isin(dates)]
        
        # 只保留需要的列
        filtered_bond_df = filtered_bond_df[['日期'] + bond_columns]
        
        return filtered_bond_df
    
    except Exception as e:
        print(f"获取国债收益率数据时出错: {str(e)}")
        return pd.DataFrame()

def merge_stock_bond_data(stock_df, bond_df):
    """合并股票和债券数据"""
    try:
        # 执行左连接
        merged_df = pd.merge(
            stock_df,
            bond_df,
            on='日期',
            how='left'
        )
        return merged_df
    
    except Exception as e:
        print(f"合并数据时出错: {str(e)}")
        return pd.DataFrame()

def export_df_to_csv(df, df_name='default_name'):
    """导出数据到CSV文件"""
    try:
        today_str = datetime.now().strftime('%Y%m%d')
        filename = f"{df_name}{today_str}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"已导出文件：{filename}")
    except Exception as e:
        print(f"导出CSV文件时出错: {str(e)}")

def main(symbol_name):
    # 1. 获取指数月末数据
    stock_month_end_df = get_month_end_stock_data(symbol_name)
    if stock_month_end_df.empty:
        print(f"未获取到{symbol_name}数据")
        return
        
    # 2. 获取国债收益率数据
    dates = stock_month_end_df['日期']
    bond_df = get_bond_yield_on_dates(dates)
    if bond_df.empty:
        print("未获取到国债收益率数据")
        return
        
    # 3. 合并数据
    merged_df = merge_stock_bond_data(stock_month_end_df, bond_df)
    if merged_df.empty:
        print("数据合并失败")
        return
        
    # 4. 导出结果
    export_df_to_csv(merged_df, f'merged_{symbol_name}_bond_data')

if __name__ == "__main__":
    symbol_name = "中证A500"  # 可以根据需要修改为其他指数
    main(symbol_name)