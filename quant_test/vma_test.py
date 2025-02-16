# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import talib
import numpy as np
from datetime import datetime, timedelta

def init(context):
    # 设置标的和订阅数据
    context.symbol = 'SZSE.300353'
    subscribe(symbols=context.symbol, frequency='1d', count=100)  # 增加到100根以确保指标计算
    
    # 设置指标参数
    context.params = {
        'ema_period': 12,     # EMA周期
        'turn_periods': 3,    # 转向判断的K线数量
        # MACD参数
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        # ADX参数
        'adx_period': 14,
        'adx_threshold': 25,
        # 仓位设置
        'open_percent': 1,     # 开仓100
        'remain_percent': 1    # 减仓100
    }
    context.price_window = []

def check_buy_signal(macd, signal, adx, pdi, mdi, context):
    """检查买入信号"""
    if len(macd) < 2 or len(signal) < 2:
        return False
        
    # MACD金叉
    macd_cross = macd[-2] <= signal[-2] and macd[-1] > signal[-1]
    
    # ADX和DI判断
    adx_condition = adx[-1] > context.params['adx_threshold'] and pdi[-1] > mdi[-1]
    
    return macd_cross and adx_condition

def get_position_info(context):
    """获取当前持仓信息"""
    positions = get_position()
    current_percent = 0
    position_detail = None
    
    if positions:
        cash = get_cash()
        
        for pos in positions:
            if pos.symbol == context.symbol:
                current_percent = pos.volume * pos.last_price / (cash.nav * 100)
                position_detail = pos
                break
    
    return current_percent, position_detail

def on_bar(context, bars):
    bar = bars[0]
    context.price_window.append(bar.close)
    if len(context.price_window) > 500:
        context.price_window = context.price_window[-500:]
    
    closes = np.array(context.price_window)
    if len(closes) >= 12:
        ema12 = talib.EMA(closes, timeperiod=12)

        if len(ema12) > 1:
            # 计算当日EMA12与昨日EMA12的差值 (当日 - 昨日)
            difference = ema12[-1] - ema12[-2]
            
            position = context.account().position(symbol=context.symbol, side=OrderSide_Buy)
            if position:
                avg_price = position.vwap  # 持仓均价
                current_price = bar.close  # 当前收盘价/最新价格
                
                # 1. 检测EMA12卖出条件
                if difference < -0.05:
                    print(f"EMA12 卖出信号: 今日({ema12[-1]:.4f}) - 昨日({ema12[-2]:.4f}) = {difference:.4f}")
                    order_target_percent(
                        symbol=context.symbol,
                        percent=0,
                        order_type=OrderType_Market,
                        position_side=PositionSide_Long
                    )
                else:
                    # 2. 若EMA12卖出条件未满足，但浮亏大于3%，执行止损
                    #   这里用当前价小于买入均价97%表示亏损大于3%
                    if current_price < avg_price * 0.97:
                        print("止损触发：当前价比买入均价亏损超3%")
                        order_target_percent(
                            symbol=context.symbol,
                            percent=0,
                            order_type=OrderType_Market,
                            position_side=PositionSide_Long
                        )
    
    print('\n' + '=' * 50)
    print(f"当前时间: {context.now}")

    try:
        # 获取历史数据
        data = context.data(symbol=context.symbol, 
                          frequency='1d',
                          count=100,
                          fields='close,open,high,low,volume,bob')
        
        if len(data) == 0:
            print("未获取到数据")
            return
            
        # 获取价格数据
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # 计算EMA12
        ema12 = talib.EMA(close, timeperiod=context.params['ema_period'])
        
        # 计算MACD
        macd, signal, hist = talib.MACD(close, 
                                      fastperiod=context.params['macd_fast'],
                                      slowperiod=context.params['macd_slow'], 
                                      signalperiod=context.params['macd_signal'])
        
        # 计算ADX
        adx = talib.ADX(high, low, close, timeperiod=context.params['adx_period'])
        pdi = talib.PLUS_DI(high, low, close, timeperiod=context.params['adx_period'])
        mdi = talib.MINUS_DI(high, low, close, timeperiod=context.params['adx_period'])
        
        # 获取当前持仓信息
        current_percent, position = get_position_info(context)
        
        # 打印持仓信息
        print("\n当前持仓信息:")
        if position:
            print(f"持仓数量: {position.volume}股")
            print(f"持仓比例: {current_percent:.2%}")
            print(f"持仓均价: {position.vwap:.2f}")
            print(f"最新价格: {position.last_price:.2f}")
            print(f"浮动盈亏: {position.fpnl:.2f}")
        else:
            print("当前无持仓")
        
        # 打印当前指标状态
        print(f"\n当前指标状态:")
        print(f"EMA12: {ema12[-1]:.2f}")
        print(f"MACD: {macd[-1]:.4f}, Signal: {signal[-1]:.4f}")
        print(f"ADX: {adx[-1]:.2f}, +DI: {pdi[-1]:.2f}, -DI: {mdi[-1]:.2f}")
        
        # 检查是否需要卖出
        if current_percent > context.params['remain_percent']:
            if check_ema_turn(ema12, periods=context.params['turn_periods']):
                print("\n触发卖出信号 - EMA12转向下跌")
                order_target_percent(symbol=context.symbol, 
                                  percent=context.params['remain_percent'],
                                  order_type=OrderType_Market,
                                  position_side=PositionSide_Long)
                return
        
        # 判断买入信号
        if check_buy_signal(macd, signal, adx, pdi, mdi, context):
            if current_percent < context.params['open_percent']:
                print("\n触发买入信号")
                print(f"MACD: {macd[-1]:.4f}, Signal: {signal[-1]:.4f}")
                print(f"ADX: {adx[-1]:.2f}, +DI: {pdi[-1]:.2f}, -DI: {mdi[-1]:.2f}")
                order_target_percent(symbol=context.symbol,
                                  percent=context.params['open_percent'],
                                  order_type=OrderType_Market,
                                  position_side=PositionSide_Long)
        
    except Exception as e:
        print(f"策略执行错误: {str(e)}")
        import traceback
        print(traceback.format_exc())

def on_order_status(context, order):
    """订单状态更新"""
    if order.status == OrderStatus_Filled:
        print(f"\n订单执行完成:")
        print(f"方向: {'买入' if order.side == OrderSide_Buy else '卖出'}")
        print(f"价格: {order.price}")
        print(f"数量: {order.volume}")
        print(f"目标仓位: {order.target_percent:.2%}")

if __name__ == '__main__':
    backtest_start_time = str(datetime.now() - timedelta(days=720))[:19]
    backtest_end_time = str(datetime.now())[:19]
    
    run(strategy_id='strategy_id',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='{{token}}',
        backtest_start_time=backtest_start_time,
        backtest_end_time=backtest_end_time,
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)