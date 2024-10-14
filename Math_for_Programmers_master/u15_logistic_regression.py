from u15_car_data import bmws, priuses
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from math import exp, log
"""
绘图函数部分
"""


def plot_function(f, xmin, xmax, **kwargs):
    """

    :param f:
    :param xmin:
    :param xmax:
    :param kwargs:
    :return:
    """
    ts = np.linspace(xmin, xmax, 1000)
    plt.plot(ts, [f(t) for t in ts], **kwargs)


def scalar_field_heatmap(f, xmin, xmax, ymin, ymax, xsteps=100, ysteps=100):
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


def plot_scalar_field(f, xmin, xmax, ymin, ymax, xsteps=100, ysteps=100, c=None,
                      cmap=cm.coolwarm, alpha=1,
                      antialiased=False, zorder=0):
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    ax = fig.add_subplot(111, projection='3d')  # 根据输入函数绘制三维坐标图

    fv = np.vectorize(f)

    # Make data.
    X = np.linspace(xmin, xmax, xsteps)
    Y = np.linspace(ymin, ymax, ysteps)
    X, Y = np.meshgrid(X, Y)
    Z = fv(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, color=c, alpha=alpha,
                           linewidth=0, antialiased=antialiased, zorder=zorder)


def plot_data(ds):
    """
    plot_data 辅助函数将获取整个汽车数据列表，并用×代表宝马，用圆点代表普锐斯
    :param ds:
    :return:
    """

    plt.scatter([d[0] for d in ds if d[2]==0], [d[1] for d in ds if d[2]==0], c='C1')
    plt.scatter([d[0] for d in ds if d[2]==1], [d[1] for d in ds if d[2]==1], c='C0', marker='x')
    plt.ylabel("Price ($)", fontsize=16)
    plt.xlabel("Odometer (mi)", fontsize=16)


"""
功能函数部分
"""


def bmw_finder(mileage, price):
    """
    分类器: bmw_finder,查看汽车的行驶里程数和价格，并给出结果表明它们是否为宝马：如果是，返回1；否则，返回0,
    :param mileage: 里程
    :param price: 价格
    :return:
    """
    if price > 25000:
        return 1  # 宝马标识
    else:
        return 0  # 普锐斯标识


def bmw_finder2(mileage, price):
    """
    分类器: bmw_finder,查看汽车的行驶里程数和价格，并给出结果表明它们是否为宝马：如果是，返回1；否则，返回0
    对比bmw_finder,在判断汽车不是宝马时过于激进,bmw_finder2将价格调整到20000,期待更好的分类效果
    :param mileage: 里程
    :param price: 价格
    :return:
    """
    if price > 20000:
        return 1  # 宝马标识
    else:
        return 0  # 普锐斯标识


def decision_boundary_classify(mileage, price):
    """
    实现分类函数:把决策边界转换为分类函数，该函数接收汽车里程数和价格，并根据该点在直线上方还是直线下方返回1或0
    :param mileage: 里程
    :param price: 价格
    :return:
    """
    if price > 21000 - 0.07 * mileage:
        return 1
    else:
        return 0


# 实现最佳分类精度的p = constant形式的决策边界需要如下两个函数(constant_price_classifier(), cutoff_accuracy())，建构建分类器函数
def constant_price_classifier(cutoff_price):
    """
    分类器函数:为任意指定的恒定临界价格建构建分类器函数。即测试车的价格高于临界价格，返回1；否则返回0。
    :param cutoff_price: 临界价格
    :return:
    """
    def c(x, p):
        if p > cutoff_price:
            return 1
        else:
            return 0
    return c


def cutoff_accuracy(cutoff_price):
    """
    辅助函数目的: 通过将constant_price_classifier(cutoff_price)得到的分类器传递给test_classify函数来衡量这个函数的准确性
    辅助函数作用: 以自动检查我们想要测试的任何价格（作为临界值）
    :param cutoff_price:
    :return:
    """
    c = constant_price_classifier(cutoff_price)
    return test_classifier(c, all_car_data)


# 创建包含mileage,price,标识的列表all_car_data, 将数据从全部数据中提取
all_car_data = []
for bmw in bmws:  # 宝马元组
    all_car_data.append((bmw.mileage, bmw.price, 1))
for prius in priuses:  # 普锐斯元组
    all_car_data.append((prius.mileage, prius.price, 0))


def test_classifier(classifier, data, verbose=False):
    """
    测试分类函数: 衡量分类器算法的效果,传入分类器效果,和测试数据集,返回分类器可以正确识别多少辆汽车,从而评估分类器的效果,并且打印真阳性、真阴性、假阳性和假阴性的数量
    :param classifier:  一个分类函数（比如bmw_finder）
    :param data:  一个用于测试的数据集
    :param verbose: 指定是否打印数据(默认是不打印)
    :return: 一个百分比值，表明可以正确识别多少辆汽车
    """
    # 真阳性、真阴性、假阳性和假阴性的数量
    true_positives = 0  # 真阳性,如果正确地识别出了宝马或普锐斯，则分别称为真阳性（true positive）或真阴性（true negative）。
    true_negatives = 0  # 真阴性,如果正确地识别出了宝马或普锐斯，则分别称为真阳性（true positive）或真阴性（true negative）。
    false_positives = 0  # 假阳性,如果它预测一辆车是宝马（返回1），但实际上是一辆普锐斯，称为假阳性（false positive）。
    false_negatives = 0  # 假阴性,如果它预测一辆车是普锐斯（返回 0），但实际上是一辆宝马，称为假阴性（false negative）。

    for mileage, price, is_bmw in data:
        predicted = classifier(mileage, price)
        if predicted and is_bmw:  # 根据该车是普锐斯还是宝马,以及分类是否正确,给其中一个计数器加1
            true_positives += 1
        elif predicted:
            false_positives += 1
        elif is_bmw:
            false_negatives += 1
        else:
            true_negatives += 1

    if verbose:  # 如果需要打印,打印每个计数器的结果
        print("true positives %f" % true_positives)
        print("true negatives %f" % true_negatives)
        print("false positives %f" % false_positives)
        print("false negatives %f" % false_negatives)

    total = true_positives + true_negatives

    return total / len(data)  # 返回正确分类的数量（真阳性或真阴性）除以数据集的长度


def make_scale(data):
    """
    辅助函数:  缩放原始汽车数据,它接收一个数字列表并返回一组缩放函数，根据列表中的最大值和最小值在0和1之间线性地缩放和还原这些数据

    :param data:
    :return:
    """
    min_val = min(data)  # 最大和最小值确定了当前数据集的范围
    max_val = max(data)

    def scale(x):
        return (x - min_val) / (max_val - min_val)  # 把min_val 和max_val 之间的数据点等比缩小到0和1之间

    def unscale(y):
        return y * (max_val - min_val) + min_val  # 把缩放后0和1之间的数据点等比还原到min_val和max_val之间
    return scale, unscale  # 如果想对这个数据集进行缩放或还原，就返回相应的函数（闭包）


price_scale, price_unscale = make_scale([x[1] for x in all_car_data])  # 返回两套函数，一套用于价格，一套用于里程数
mileage_scale, mileage_unscale = make_scale([x[0] for x in all_car_data])


def sigmoid(x):
    """
    最基本的logistic函数如下，通常称为sigmoid函数: σ(x) = 1 / 1 + e**(-x)
    :param x:
    :return:
    """
    return 1 / (1+exp(-x))


def f(x, p):
    """
    f(x, p) = p − ax − b, 返回值可能很大，可能是正数，也可能是负数。0表示它处于宝马和普锐斯之间的边界上
    :param x:里程值
    :param p:价格值
    :return:返回一个数，该数衡量这些值在多大程度上更像宝马而不是普锐斯
    """
    return p + 0.35 * x - 0.56


def l(x, p):
    """
    要将 f(x, p)的输出调整到预期范围内，只需通过sigmoid函数 σ(x)即可。
    就是说，我们要的函数是 σ(f(x, p)),
    目的: 用L(x, p)这样的函数来拟合数据
    :param x: 里程数
    :param p: 价格
    :return:
    """
    return sigmoid(f(x, p))


def make_logistic(a, b, c):
    """
    二维的logistic函数:
    :param a:
    :param b:
    :param c:
    :return: 相应的 logistic函数 L(x, p) = σ(ax + bp − c)
    """
    def l(x, p):
        return sigmoid(a*x + b*p - c)
    return l


def simple_logistic_cost(a, b, c):
    """
    代价函数: 衡量函数L的误差或代价，一个简单的方法是找出它与正确值（0或1）之间的误差。
    如果把所有误差加起来，得到的总值就表明函数L(x, p)与数据集的差距,
    这个代价函数很好地报告了误差，但不足以使梯度下降收敛到a、b和c的最佳值。
    :param a:
    :param b:
    :param c:
    :return:
    """
    l = make_logistic(a, b, c)
    errors = [abs(is_bmw-l(x, p))
              for x, p, is_bmw in scaled_car_data]
    return sum(errors)


def point_cost(l, x, p, is_bmw):
    """
    待理解2024.09.16
    :param l:
    :param x:
    :param p:
    :param is_bmw:
    :return:
    """
    wrong = 1 - is_bmw
    return -log(abs(wrong - l(x, p)))


"""
测试部分
"""
if __name__ == "__main__":
    # 测试分类函数逻辑,分类器没有返回假阳性，也就是说它始终可以正确地识别汽车何时不是宝马。因为它判断大多数汽车不是宝马，但其中有很多确实是宝马！
    # print(test_classifier(bmw_finder, all_car_data, verbose=True))
    # bmw_finder2,将成功率提高到了73.5%
    # print(test_classifier(bmw_finder2, all_car_data, verbose=True))
    # 测试decision_boundary_classify分类函数效果
    # print(test_classifier(decision_boundary_classify, all_car_data))

    # 数据集中所有汽车的价格与里程数的关系图，宝马用×表示，而普锐斯用圆点表示
    # plot_data(all_car_data)
    # 显示绘制了汽车数据的决策线
    # plot_function(lambda x: 25000, 0, 200000, c='k')
    # 让临界价格取决于里程数。在几何上，这意味着绘制一条向下倾斜的直线,由函数p(x) = 21 000 − 0.07 · x给出，其中p为价格，x为里程
    # plot_function(lambda x: 21000 - 0.07 * x, 0, 200000, c='k')

    # 最佳临界价格在价格列表之中。只要检查每个价格,关键参数key让我们可以选择用什么函数来进行最大化
    # all_prices = [price for (mileage, price, is_bmw) in all_car_data]
    # 数据集中找到最好的临界价格，可以通过cutoff_accuracy函数来进行最大化
    # print(max(all_prices, key=cutoff_accuracy))
    # 对最佳临界价格测试,测试 test_classifier 分类函数效果，准确率为79.5%
    # print(test_classifier(constant_price_classifier(17998.0), all_car_data))

    # 里程和价格数据按比例缩放，使所有数值都在0和1之间。该图看起来和以前一样，但数值误差风险降低了
    scaled_car_data = [(mileage_scale(mileage), price_scale(price), is_bmw)
                       for mileage, price, is_bmw in all_car_data]
    # plot_data(scaled_car_data)

    # 查看带有数据的f(x, p) = p − ax − b的热力图（见图15-9）。当a = –0.35、b = 0.56时，函数为f(x, p) = p− 0.35 · x − 0.56
    # 热图和决策边界图，展示出亮值（正“宝马性”）位于决策边界之上，暗值（负“宝马性”）位于决策边界之下
    # scalar_field_heatmap(lambda x, p: p + 0.35 * x - 0.56, 0, 1, 0, 1)
    # plt.ylabel('Price', fontsize=16)
    # plt.xlabel('Mileage', fontsize=16)
    # plot_function(lambda x: 0.56 - 0.35 * x, 0, 1, c='k')

    # sigmoid函数σ(x)的曲线图
    # plot_function(sigmoid, -5, 5)

    # 函数L(x, p)的值 和 f(x, p)的值对比二维图
    # scalar_field_heatmap(l, 0, 1, 0, 1)
    # # plot_data(scaled_car_data,white=True)
    # plt.ylabel('Price', fontsize=16)
    # plt.xlabel('Mileage', fontsize=16)
    # plot_function(lambda x: 0.56 - 0.35 * x, 0, 1, c='k')
    # 函数 f(x, p)的值三维图
    plot_scalar_field(l, -5, 5, -5, 5)
    # 函数 L(x, p)的值三维图
    plot_scalar_field(f, -5, 5, -5, 5)
    # 结论:如果用0或1表示汽车的类型,那么函数L(x, p)的值实际上会接近这些数，而f(x, p)的值则会走向正无穷和负无穷

    # 显式调用图形界面
    plt.show()



