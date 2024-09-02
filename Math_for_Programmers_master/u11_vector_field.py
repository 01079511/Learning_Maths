import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def f(x, y):
    """
    定义二维矢量函数,描述一个 向量场 ,其中每个点(x, y)的向量由(-2*y, x)表示
    :param x:"'
    :param y:
    :return: 二维向量 (-2 * y, x)
    """
    # return -2 * y, x
    # return -x, -y
    return 0.5 * (x**2 + y**2)


def u(x, y):
    """
    利用U(x, y) = 1/2(x**2 + y**2)来定义一个标量场
    """
    return 0.5 * (x**2 + y**2)


def plot_vector_field(f, xmin, xmax, ymin, ymax, xstep=1, ystep=1):
    """
    绘制向量场函数
    np.meshgrid: 生成两个二维数组X和Y,表示给定范围内所有(x,y)的坐标网格
    np.vectorize: 将标量函数f(x, y)转化为可接受数组输入的函数,U 和 V 表示网格上每个(x,y)点处向量的x和y分量
    plt.quiver函数绘制二维向量场X和Y是网格坐标，U和V是对应处的向量分量
    plt.gcf()获取当前Figure对象, 并设置图形大小7*7
    """
    X,  Y = np.meshgrid(np.arange(xmin, xmax, xstep), np.arange(ymin, ymax, ystep))
    U = np.vectorize(lambda x, y: f(x, y)[0])(X, Y)
    V = np.vectorize(lambda x, y: f(x, y)[1])(X, Y)
    plt.quiver(X, Y, U, V, color='red')
    fig = plt.gcf()
    fig.set_size_inches(7, 7)


def plot_scalar_field(f, xmin, xmax, ymin, ymax, xstep=0.25, ystep=0.25, c=None,
                      cmap=cm.coolwarm, alpha=1, antialiased=False):
    """
    plot_scalar_field 函数接收一个定义标量场的函数，以及x和y的边界，并绘制出代表该场的三维点的表面;
    随着距离原点(0，0)越来越远，势能也会越来越大。在所有的径向方向上，图形的高度都会增加，即U值增大
    """
    fig = plt.figure()
    fig.set_size_inches(7, 7)
    ax = fig.add_subplot(111, projection='3d')  # 使用add_subplot 创建3D轴对象,111表示1*1网格中第一个子图

    fv = np.vectorize(f)

    # Make data.
    X = np.arange(xmin, xmax, xstep)
    Y = np.arange(ymin, ymax, ystep)
    X, Y = np.meshgrid(X, Y)
    Z = fv(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, color=c, alpha=alpha,
                           linewidth=0, antialiased=antialiased)


def scalar_field_heatmap(f, xmin, xmax, ymin, ymax, xstep=0.1, ystep=0.1):
    """
    scalar_field_heatmap函数来绘制势能函数(热图)
    靠近(0, 0)的地方，颜色较深，意味着U(x, y)的值较小。在图的边缘，颜色较浅，意味着U(x, y)的值较大
    """
    fig = plt.figure()
    fig.set_size_inches(7, 7)

    fv = np.vectorize(f)

    X = np.arange(xmin, xmax, xstep)
    Y = np.arange(ymin, ymax, ystep)
    X, Y = np.meshgrid(X, Y)

    # https://stackoverflow.com/a/54088910/1704140
    z = fv(X, Y)

    #     # x and y are bounds, so z should be the value *inside* those bounds.
    #     # Therefore, remove the last value from the z array.
    #     z = z[:-1, :-1]
    #     z_min, z_max = -z.min(), z.max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(X, Y, z, cmap='plasma')
    # set the limits of the plot to the limits of the data
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    fig.colorbar(c, ax=ax)
    plt.xlabel('x')
    plt.ylabel('y')


def scalar_field_contour(f, xmin, xmax, ymin, ymax, levels=None):
    """
    标量场绘制成等高线图,可以将其解释为U(x, y)在离原点越远的地方越陡峭
    """
    fv = np.vectorize(f)

    X = np.arange(xmin, xmax, 0.1)
    Y = np.arange(ymin, ymax, 0.1)
    X, Y = np.meshgrid(X, Y)

    # https://stackoverflow.com/a/54088910/1704140
    Z = fv(X, Y)

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels=levels)
    ax.clabel(CS, inline=1, fontsize=10, fmt='%1.1f')
    plt.xlabel('x')
    plt.ylabel('y')
    fig.set_size_inches(7, 7)


# 从-5到5的范围,绘制函数f(x, y)的向量场
# plot_vector_field(f, -5, 5, -5, 5)
# plot_scalar_field(u, -5, 5, -5, 5)
# scalar_field_heatmap(u, -5, 5, -5, 5)
scalar_field_contour(u, -10, 10, -10, 10, levels=[10, 20, 30, 40, 50, 60])

# 显示调用plt.show()打开图形界面
plt.show()
