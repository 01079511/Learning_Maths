import matplotlib.pyplot as plt
import numpy as np


def plot_function(f, tmin, tmax, tlabel=None, xlabel=None, axes=False, **kwargs):
    """
    绘制函数f在区间[tmin,tmax]的图像
    :param f:目标函数
    :param tmin: t1
    :param tmax: t2
    :param tlabel:如有,则设置x轴标签
    :param xlabel:如有,则设置y轴标签
    :param axes:如有,则绘制轴线
    :param **kwargs:用于接收任意数量的关键字参数,参数会被作为字典传递给函数,key=参数名,vul=参数值
    :return:
    """
    ts = np.linspace(tmin, tmax, 1000)  # 生成ts数组,从tmin->tmax的1000个等间隔数组ts
    if tlabel:
        plt.xlabel(tlabel, fontsize=18)
    if xlabel:
        plt.ylabel(xlabel, fontsize=18)
    # 绘制函数图像,利用列表生成式[f(t) for t in ts]计算每个ts值对应的f(t),**kwargs用于传递其他绘图参数
    plt.plot(ts, [f(t) for t in ts], **kwargs)
    if axes:
        # 时间差
        total_t = tmax-tmin
        # 绘制x轴范围是[tmin-total_t/10, tmax+total_t/1],比原始最大值最小值多出10%,
        # y轴在[0, 0],
        # c='k'线条颜色为黑色k,
        # linewidth=1线条宽度是1
        plt.plot([tmin-total_t/10, tmax+total_t/10], [0, 0], c='k', linewidth=1)
        #  设置x轴的显示范围,即起点和终点,[tmin-total_t/10, tmax+total_t/1],比原始最大值最小值多出10%,
        plt.xlim(tmin-total_t/10, tmax+total_t/10)
        # 调用plt.ylim()返回当前y轴上下限,赋值给xmin, xmax
        xmin, xmax = plt.ylim()
        plt.plot([0, 0], [xmin, xmax], c='k', linewidth=1)
        plt.ylim(xmin, xmax)


def plot_volume(f, tmin, tmax, axes=False, **kwargs):
    # 绘制体积随时间变化的图像,标注x/y轴标识
    plot_function(f, tmin, tmax,
                  tlabel="time (hr)", xlabel="volume (bbl)",
                  axes=axes, **kwargs)


def plot_flow_rate(f, tmin, tmax, axes=False, **kwargs):
    # 绘制流量随时间变化的图像,标注x/y轴标识
    plot_function(f, tmin, tmax,
                  tlabel="time (hr)", xlabel="flow rate (bbl/hr)",
                  axes=axes, **kwargs)


def volume(t):
    # 定义体积随时间变化的函数(直接给出)
    return (t-4)**3 / 64 + 3.3


def flow_rate(t):
    # 定义流量随时间变化的函数(直接给出)
    return 3*(t-4)**2 / 64


def decreasing_volume(t):
    """
    绘制了油箱中石油体积随时间递减的曲线
    :param t:时间
    :return:随时间递减后的体积
    """
    if t < 5:
        return 10 - (t**2)/5
    else:
        return 0.2*(10-t)**2


def average_flow_rate(v, t1, t2):
    """
    根据体积计算平均流速
    :param v:体积函数v
    :param t1:开始时间t1
    :param t2:结束时间t2
    :return:返回一个数，表示在这个时间段内进入油箱的平均流速
    """
    return (v(t2) - v(t1))/(t2 - t1)


def secant_line(f, x1, x2):
    """
    割线
    :param f:收函数f(x)
    :param x1:值
    :param x2:值
    :return:一个表示随时间变化割线的新函数
    """
    def line(x):
        return f(x1) + (x-x1) * (f(x2)-f(x1))/(x2-x1)
    return line


def plot_secant(f, x1, x2, color='k'):
    """
    结合secant_line,实现两个给定点之间绘制函数f的割线
    :param f: 用于接受函数
    :param x1:
    :param x2:
    :param color:颜色
    :return:
    """
    line = secant_line(f, x1, x2)
    plot_function(line, x1, x2, c=color)
    plt.scatter([x1, x2], [f(x1), f(x2)], c=color)  # 绘制散点图


def interval_flow_rates(v, t1, t2, dt):
    """
    计算不同时间段内的平均流速，来近似得到流速随时间变化的函数(实例10小时内按小时分段)
    :param v:体积函数
    :param t1:开始时间
    :param t2:结束时间
    :param dt:间隔
    :return:包括时间和流速对的列表
    """
    # 对于每一个间隔的开始时间 t，找到从t到t+dt的平均流速（我们要的是t及其对应速率的列表）
    return [(t, average_flow_rate(v, t, t+dt)) for t in np.arange(t1, t2, dt)]


def plot_interval_flow_rates(volume, t1, t2, dt):
    """
    利用scatter 函数来快速绘制表示这些流速随时间变化的图
    :param v:体积函数
    :param t1:开始时间
    :param t2:结束时间
    :param dt:间隔
    :return:将结果传给scatter函数
    """
    series = interval_flow_rates(volume, t1, t2, dt)
    times = [t for (t, _) in series]
    rates = [q for (_, q) in series]
    plt.scatter(times, rates)


def linear_volume_function(t):
    """
    实现linear_volume_function函数，并画出流速随时间的变化图，以说明流速是恒定的
    数学形式是： V(t) = at + b, 其中a和b为常数
    :param t:
    :return:
    """
    return 5 * t + 3


def instantaneous_flow_rate(v, t, digits=6):
    """
    瞬时流速函数,(在微积分中称为体积函数的导数),实现从一个体积函数开始，产生一个相应的流速函数
    存在问题:Python不能通过"目测"几条小割线的斜率来决定它们会收敛到什么数
    解决方式:将割线不断缩短并计算其斜率，直到斜率数值稳定在某个固定的小数位上
    :param v:体积函数
    :param t:单一的时间点t
    :param digits:
    :return:获取某一时刻的瞬时流速
    """
    tolerance = 10 ** (-digits)  # 如果两个数相差小于容差(tolerance)10的–d次方，那么它们精确到小数点后d位是相同的
    h = 1
    approx = average_flow_rate(v, t-h, t+h)  # 首先计算目标点t 两侧h = 1个单位间隔上的割线斜率
    for i in range(0, 2*digits):  # 作为一个粗略的近似值，我们在两次迭代之后放弃继续计算
        h = h / 10  # 在每一步，将围绕点t的间隔缩小10倍
        next_approx = average_flow_rate(v, t-h, t+h)  # 计算这个新间隔内割线的斜率
        if abs(next_approx - approx) < tolerance:  # 如果最后两个近似值相差小于容差，则将四舍五入后的结果返回
            return round(next_approx, digits)
        else:
            approx = next_approx
    raise Exception("Derivative did not converge")  # 如果超过了最大的迭代次数，就表示程序没有收敛到一个结果


def get_flow_rate_function(v):
    """
    柯里化(currying)瞬时流速函数
    目的:实现与源代码中flow_rate函数行为类似的函数，
    也就是接收一个时间变量并返回一个流速值的函数，对instantaneous_flow_rate函数进行柯里化
    :param v:体积函数(v)
    :return:流速函数
    """
    def flow_rate_function(t):
        instantaneous_flow_rate(v, t)
    return flow_rate_function


print('volume(4)={0:.2f},\n'
      'volume(9)={1:.2f},\n'
      'average_flow_rate(volume,4,9)={2:.2f}\n'.format(volume(4),
                                                 volume(9),
                                                 average_flow_rate(volume, 4, 9)))
# 打印interval_flow_rates(v, t1, t2, dt)
# print(interval_flow_rates(volume, 0, 10, 1))

# 体积随时间变化曲线,递增
# plot_volume(volume, 0, 10)
# 体积随时间变化曲线,递减
# plot_volume(decreasing_volume, 0, 10)
# flow_rate 函数表示的是任意时间点上流速的瞬时值。
# plot_flow_rate(flow_rate, 0, 10)
# 割线
# plot_secant(volume, 4, 8)

# interval_flow_rates所产生数据的散点图
# plot_interval_flow_rates(volume, 0, 10, 1)
# 绘制decreasing_volume 函数流速随时间的变化图。什么时候其流速为最小值？也就是说，石油什么时候离开油箱的速度最快？
# plot_interval_flow_rates(decreasing_volume, 0, 10, 0.5)
# linear_volume_function函数,画出流速随时间的变化图,体现流速是恒定的 ==> 线性体积函数,流速不随时间变化
# plot_interval_flow_rates(linear_volume_function, 0, 10, 0.25)

# 测试instantaneous_flow_rate(volume,1)
# print("instantaneous_flow_rate(volume,1)={}\n".format(instantaneous_flow_rate(volume, 1)))

# get_flow_rate_function(v)的输出是一个函数，与源代码中的flow_rate相同。画图测试
# plot_function(flow_rate,0,10)
# plot_function(get_flow_rate_function(volume), 0, 10)

# 画图
plt.show()
plt.close()
