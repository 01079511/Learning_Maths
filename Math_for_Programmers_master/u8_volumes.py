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
    :return:一个表示随时间变化割线的新函数line
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
        return instantaneous_flow_rate(v, t)
    return flow_rate_function


def sign(x):
    """
    证明该函数在x = 0处不存在导数,在时间间隔变得越来越小时，割线的斜率会越来越大，而不是收敛在一个值上
    average_flow_rate(sign, -0.1, 0.1) => 10.0
    average_flow_rate(sign, -0.000001, 0.000001) => 1000000.0
    :param x:
    :return:接收x,传入目标函数,如average_flow_rate
    """
    return x / abs(x)


def small_volume_change(q, t, dt):
    """
    体积变化的近似值:
    体积的变化约等于 流速*经过的时间
    :param q:流速函数q
    :param t:输入时间t
    :param dt:间隔dt
    :return:体积的变化
    """
    return q(t) * dt


def volume_change(q, t1, t2, dt):
    """
    目的:因为在短间隔内可以得到很好的体积变化近似值，所以我们可以把它们累加起来，得到较长间隔内的体积变化。
    问题:
    如果,使宽度为0.5小时的8个矩形超过了3.9小时对应的范围，我们还是使用所有8个矩形的面积来进行计算！
    为了让计算精确，应该将最后一个矩形的时间间隔缩短到0.4小时。
    这个矩形从第7个间隔末的3.5小时持续到结束时间3.9小时，不再覆盖更多范围,需要迭代 volume_change
    :param q:流速函数q
    :param t1:开始时间
    :param t2:结束时间
    :param dt:间隔
    :return:dt间隔内的体积总变化
    """
    return sum(small_volume_change(q, t, dt)
               for t in np.arange(t1, t2, dt))  # 调用 np.arrange (t1,t2,dt)可以给我们提供一个从t1到t2的、增量为dt的时间数组


def approximate_volume(q, v0, dt, T):
    """
    实现: T时刻的体积 = (0时刻的体积) + (从0时刻到T时刻的体积变化值)
    :param q:流速函数q
    :param v0: 0时刻油箱中的体积
    :param dt: 间隔
    :param T:T时刻
    :return:从0时刻到T时刻的体积变化值
    """
    return v0 + volume_change(q, 0, T, dt)


def approximate_volume_function(q, v0, dt):
    """
    柯里化 approximate_volume
    :param q: 流速函数q
    :param v0:0时刻油箱中的体积
    :param dt:间隔
    :return:参数T作为输入的函数
    """
    def volume_function(T):
        return approximate_volume(q, v0, dt, T)
    return volume_function


def get_volume_function(q, v0, digits=6):
    """
    对于任意时间点t,用越来越小的dt值重复计算volume_change(q,0,t,dt),直到输出稳定在容差范围内.
    这看起来很像之前对导数进行反复逼近(instantaneous_flow_rate),直到它稳定下来的过程。
    :param q:流速函数q
    :param v0:0时刻油箱中的体积
    :param digits:任意容差
    :return:
    """
    def volume_function(T):
        tolerance = 10 ** (-digits)
        dt = 1
        approx = v0 + volume_change(q, 0, T, dt)
        for i in range(0, digits*2):
            dt = dt / 10
            next_approx = v0 + volume_change(q, 0, T, dt)
            if abs(next_approx - approx) < tolerance:
                return round(next_approx,digits)
            else:
                approx = next_approx
        raise Exception("Did not converge!")
    return volume_function


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

# 体积函数和流速函数对比,间隔大则误差大
# print(small_volume_change(flow_rate, 2, 1))
# 0.1875
# print(volume(3) - volume(2))
# 0.109375

# 分拆极小间隔求和, 实现缩短误差
# print(volume_change(flow_rate, 0, 10, 0.1))
# 4.32890625
# print(volume(10) - volume(0))
# 4.375
# 实践:前6小时内大约有多少石油被添加到油箱中
# print(volume_change(flow_rate,0,6,0.01))
# 1.1278171874999996
# 最后4小时内大约有多少石油被添加到油箱中
# print(volume_change(flow_rate,6,10,0.01))
# 3.2425031249999257

# 计算在任意时间点上油箱中的石油总体积
# plot_function(approximate_volume_function(flow_rate, 2.3, 0.01), 0, 10)
# plot_function(volume, 0, 10)

# 随着dt值越来越小，体积近似值趋近于volume 函数的精确值。它所趋近的结果称为流速的积分
# v = get_volume_function(flow_rate, 2.3, digits=3)
# print(v(1))
# v = get_volume_function(flow_rate, 2.3, digits=6)
# print(v(1))

# 画图
plt.show()
plt.close()
# 08.19