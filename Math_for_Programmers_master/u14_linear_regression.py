from math import exp
import numpy as np
import matplotlib.pyplot as plt
from u14_car_data import priuses
from u7_vectors import length

test_data = [
     (-1.0, -2.0137862606487387),
     (-0.9, -1.7730222478628337),
     (-0.8, -1.5510125944820812),
     (-0.7, -1.6071832453434687),
     (-0.6, -0.7530149734137868),
     (-0.5, -1.4185018340443283),
     (-0.4, -0.6055579756271128),
     (-0.3, -1.0067254915961406),
     (-0.2, -0.4382360549665138),
     (-0.1, -0.17621952751051906),
     (0.0, -0.12218090884626329),
     (0.1, 0.07428573423209717),
     (0.2, 0.4268795998864943),
     (0.3, 0.7254661223608084),
     (0.4, 0.04798697977420063),
     (0.5, 1.1578103735448106),
     (0.6, 1.5684111061340824),
     (0.7, 1.157745051031345),
     (0.8, 2.1744401978240675),
     (0.9, 1.6380001974121732),
     (1.0, 2.538951262545233)
]

"""
绘图函数区域
"""


def plot_function(f, xmin, xmax, **kwargs):
    ts = np.linspace(xmin, xmax, 1000)
    plt.plot(ts, [f(t) for t in ts], **kwargs)


def draw_cost(h, points):
    """
    误差值绘图函数: 误差值是实际y值与预测值f(x)之间的差,表示为点(x, y)到f的垂直距离
    :param h:函数
    :param points:点(x, y)的集合
    :return:
    """
    xs = [t[0] for t in points]
    ys = [t[1] for t in points]
    plt.scatter(xs, ys)
    plot_function(h, min(xs), max(xs), c='k')
    for (x, y) in points:
        plt.plot([x, x], [y, h(x)], c='k')  # 展示误差值是函数f(x)与实际值y之间的差


def draw_square_cost(h, points):
    """
    绘图实现: 函数与数据集之间的误差平方和的图
    :param h:函数
    :param points:点(x, y)的集合
    :return:
    """
    xs = [t[0] for t in points]
    ys = [t[1] for t in points]
    plt.scatter(xs, ys)
    plot_function(h, min(xs), max(xs), c='k')
    for (x, y) in points:
        e = abs(y - h(x))
        plt.plot([x, x], [y, h(x)], c='r')
        plt.fill([x, x, x+e, x+e], [h(x), y, y, h(x)], c='r', alpha=0.5)


def plot_mileage_price(cars):
    """
    mileages,prices散点图
    :param cars:
    :return:
    """
    prices = [c.price for c in cars]
    mileages = [c.mileage for c in cars]
    plt.scatter(mileages, prices, alpha=0.5)
    plt.ylabel("Price ($)", fontsize=16)
    plt.xlabel("Odometer (mi)", fontsize=16)


def scalar_field_heatmap(f, xmin, xmax, ymin, ymax, xsteps=100, ysteps=100):
    """
    绘制热力图
    :param f:
    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :param xsteps:
    :param ysteps:
    :return:
    """
    fig = plt.figure()
    fig.set_size_inches(7, 7)
    fv = np.vectorize(f)
    X = np.linspace(xmin, xmax, xsteps)
    Y = np.linspace(ymin, ymax, ysteps)
    X, Y = np.meshgrid(X, Y)
    z = fv(X, Y)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, z, cmap='plasma')
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    fig.colorbar(c, ax=ax)


"""
功能区域
"""


# 线性函数拟合
def p(x):
    return a * x + b


# 利用梯度寻找上坡方向,第一步是要能够近似地计算出任意一点的梯度。为此，可使用第9章中介绍的方法：取短割线的斜率。
def secant_slope(f, xmin, xmax):
    return (f(xmax) - f(xmin)) / (xmax - xmin)  # 求x值在xmin和xmax之间的割线f(x)的斜率


def approx_derivative(f, x, dx=1e-6):
    return secant_slope(f, x - dx, x + dx)  # 近似导数是x − 10**(−6)和 x + 10**(−6)之间的一条割线


def approx_gradient(f, x0, y0, dx=1e-6):
    """
    为了求函数f(x, y)在点(x0, y0)处的近似偏导数，可以固定x = x0，求相对于y的导数；
    或者固定y = y0，求相对于x的导数。
    在(x0, y0)处的偏导数∂f/∂x是f(x, y0)在x = x0时相对于x的普通导数。
    同样，在y = y0时，偏导数∂f/∂y是f(x0, y)相对于y的普通导数。
    梯度是这些偏导数的向量（元组）。
    :param f:
    :param x0:
    :param y0:
    :param dx:
    :return:
    """
    partial_x = approx_derivative(lambda x: f(x, y0), x0, dx=dx)
    partial_y = approx_derivative(lambda y: f(x0, y), y0, dx=dx)
    return (partial_x, partial_y)


def gradient_descent(f, xstart, ystart, tolerance=1e-6):
    """
    梯度下降:在θϕ平面上跟踪梯度下降的轨迹。这与通过欧拉方法迭代跟踪时间和位置值的方式类似。(类似u12 gradient_ascent_points())
    :param f:
    :param xstart:
    :param ystart:
    :param tolerance:
    :return:
    """
    x = xstart
    y = ystart
    grad = approx_gradient(f, x, y)
    while length(grad) > tolerance:
        x -= 0.01 * grad[0]
        y -= 0.01 * grad[1]
        grad = approx_gradient(f, x, y)
    return x, y


def sum_error(f, data):
    """
    代价函数: 把数据集里每一个(x, y)对应的从f(x)到y的距离加起来.(假设目标函数是线性函数)
    :param f: 目标函数
    :param data: 数据集
    :return:
    """
    errors = [abs(f(x) - y) for (x, y) in data]
    return sum(errors)


def sum_squared_error(f, data):
    """
    代价函数: 实践中会使用将所有误差的平方相加的代价函数,
    因为平方距离函数是平滑的，可以用导数来最小化它；
    而绝对值函数不平滑，不是处处可导的;
    原理: 给定一个测试函数 f(x)，可以查看每一个(x, y)对，并将(f(x) – y)**2的值加到代价上
    :param f: 目标函数
    :param data: 数据集
    :return:
    """
    squared_errors = [(f(x) - y)**2 for (x, y) in data]
    return sum(squared_errors)


# 代表数据(mileage, price)的数对列表
prius_mileage_price = [(p.mileage, p.price) for p in priuses]


def test_data_coefficient_cost(a):
    """
    要绘制的函数需要两个数a和b，并返回一个数，即函数p(x) = ax + b的代价,
    作为热身，可以尝试将函数f(x) = ax拟合到之前使用的test_data数据集上(只需要调整一个参数)
    :param a: 取参数a（斜率）
    :return:  通过的sum_squared_error(),返回f(x) = ax的代价
    """
    def f(x):
        return a * x
    return sum_squared_error(f, test_data)


def coefficient_cost(a, b):
    """
    根据sum_squared_error评估效果(为了评估系数a和b的不同选择),
    类似于test_data_coefficient_cost 函数，只是有两个参数，而且使用不同的数据集
    :param a:
    :param b:
    :return: sum_squared_error()
    """
    def p(x):
        return a * x + b
    return sum_squared_error(p, prius_mileage_price)


def scaled_cost_function(c, d):
    """
    缩放数据:梯度下降前的准备(折旧率在–1和0之间，价格以万美元为单位，而代价函数返回的结果以千亿为单位,
    用步长dx为10**(–6)来计算导数的近似值,因为这些数的量级相差很大，所以直接运行梯度下降法会产生很大的数值误差)
    逻辑:将代价函数的结果除以10**13，并用c和d来表示，就得到了一个新的代价函数，其输入和输出的绝对值都在0和1之间。
        这个缩放数据的方法其实有些落后,实践在机器学习文献中通常叫作——特征缩放.
    :param c:
    :param d:
    :return:为了得到最佳拟合线，可以根据直觉对a和b设置保守的边界(该函数仅用于示范)
    """
    return coefficient_cost(0.5*c, 50000*d) / 1e13


# 非线性函数拟合
def exp_coefficient_cost(q, r):
    """
    p(x) = q*e^rx的最佳拟合函数，并使它最小化汽车数据的误差平方和,
    首先,取系数q和r并返回相应函数的代价
    :param q:
    :param r:
    :return:
    """
    def f(x):
        return q * exp(r*x)  # Python的exp函数可以计算指数函数e^x
    return sum_squared_error(f, prius_mileage_price)


def scaled_exp_coefficient_cost(s, t):
    """
    数据缩放:假设一辆车在行驶了最初的10 000英里后，价格降低到原来的1/e，即原价的约36%。这样r = 10^–4。

    :param s:
    :param t:
    :return:
    """
    return exp_coefficient_cost(30000*s, 1e-4*t) / 1e11



"""
测试区域
"""
if __name__ == "__main__":
    # 一组随机生成的test_data数据集的散点图，有意保持在接近直线f(x) = 2x的地方
    # plt.scatter([t[0] for t in test_data], [t[1] for t in test_data])
    # plot_function(lambda x: 2*x, -1, 1, c='k')

    # 展示误差值是函数f(x)与实际值y之间的差
    # draw_cost(lambda x: 2 * x, test_data)

    # 展示函数与数据集之间的误差平方和的图
    # draw_square_cost(lambda x: 2 * x, test_data)

    # test_data_coefficient_cost(a) 代价与斜率的关系
    some_slopes = [-1, 0, 1, 2, 3]
    # plt.scatter(some_slopes, [test_data_coefficient_cost(a) for a in some_slopes])
    # plt.ylabel("cost", fontsize=16)
    # plt.xlabel("a", fontsize=16)
    # 对应的直线和test_data
    # plt.scatter([t[0] for t in test_data], [t[1] for t in test_data])
    # for a in some_slopes:
    #     plot_function(lambda x: a * x, -1, 1, c='k')
    # plt.ylabel("y", fontsize=16)
    # plt.xlabel("x", fontsize=16)
    # 最佳拟合:通过代价与斜率a的关系图，显示了不同斜率值的拟合质量,通过原点的直线在斜率大约为2时得到最低的代价，此时为最佳拟合
    # plot_function(test_data_coefficient_cost, -5, 5)
    # plt.ylabel("cost", fontsize=16)
    # plt.xlabel("a", fontsize=16)

    # 系数对构成的二维空间(a, b),下图显示了ab平面上的两个点
    # a_slopes, b_slopes = [0.05, -0.2], [5000, 25000]
    # plt.scatter(a_slopes, b_slopes)
    # plt.ylabel("b", fontsize=20)
    # plt.xlabel("a", fontsize=20)
    # 由系数对构成的二维空间(a, b)它们对应的直线
    # plot_mileage_price(priuses)
    # plot_function(lambda x: 25000 - 0.20 * x, 0, 125000, c='k')
    # plot_function(lambda x: 5000 + 0.05 * x, 0, 350000, c='k')
    # 通过scalar_field_heatmap,绘制热力图对coefficient_cost函数计算(a, b)构成线性函数的空间 sum_squared_error的p(x) = ax + b相对于汽车数据的误差平方和
    # 当(a, b)取其极端值时，代价函数很高,弊端:热力图不能直观地看出代价是否存在一个最小值，或者最小值到底在哪里,改进: 梯度下降法
    # scalar_field_heatmap(coefficient_cost, -0.5, 0.5, -50000, 50000)
    # plt.ylabel("b", fontsize=16)
    # plt.xlabel("a", fontsize=16)

    # 梯度优化的函数是 scaled_cost_function,可以期望最小值出现在点(c, d)处,其中|c| < 1,|d| < 1。因为最优的 c和 d离原点很近,所以可以从(0, 0)开始梯度下降
    c, d = gradient_descent(scaled_cost_function, 0, 0)
    # print(c, d)
    # 找到a和b，需要将c和d乘以各自的系数
    a = 0.5 * c
    b = 50000 * d
    # p(x) = −0.0606 * x + 15700
    # 拟合度对比
    print(coefficient_cost(a, b))
    # 梯度优化有的线性方程在数据集中
    # plot_mileage_price(priuses)
    # plot_function(p, 0, 250000, c='k')

    # q和r分别缩放为s和t之后的代价函数
    # scalar_field_heatmap(scaled_exp_coefficient_cost, 0, 1, -1, 0)
    s, t = gradient_descent(scaled_exp_coefficient_cost, 0, 0)
    q, r = 30000*s, 1e-4*t
    print(q, r)  # p(x) = 18700 * e^(−0.000 007 68 * x)
    print(coefficient_cost(a, b))
    # 显示了实际价格的数据,梯度优化有的线性方程在数据集中
    plot_mileage_price(priuses)
    q, r = (16133.220556990309, -5.951793936498175e-06)
    plot_function(lambda x: q * exp(r * x), 0, 375000, c='k')
    print(exp_coefficient_cost(q, r))  # 可以说这个模型甚至比前面的线性模型更好，因为它产生的误差平方和更小。这意味着按照代价函数的衡量方式，它能（稍微）更好地拟合数据

    # 显式调用图形界面
    plt.show()
