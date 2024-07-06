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
    plot_function(f, tmin, tmax, tlabel="time (hr)", xlabel="volume (bbl)", axes=axes, **kwargs)


def plot_flow_rate(f, tmin, tmax, axes=False, **kwargs):
    plot_function(f, tmin, tmax, tlabel="time (hr)", xlabel="flow rate (bbl/hr)", axes=axes, **kwargs)


def volume(t):
    return (t-4)**3 / 64 + 3.3


def flow_rate(t):
    return 3*(t-4)**2 / 64


plot_volume(volume, 0, 10)
plt.show()
plt.close()

def average_flow_rate(v, t1, t2):
    return (v(t2) - v(t1))/(t2 - t1)


print('volume(4)={0:.2f},\n'
      'volume(9)={1:.2f},\n'
      'average_flow_rate(volume,4,9)={2:.2f}'.format(volume(4),
                                                 volume(9),
                                                 average_flow_rate(volume, 4, 9)))

