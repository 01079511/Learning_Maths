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
    fig = plt.figure()
    fig.set_size_inches(7, 7)
    ax = fig.gca(projection='3d')

    fv = np.vectorize(f)

    # Make data.
    X = np.arange(xmin, xmax, xstep)
    Y = np.arange(ymin, ymax, ystep)
    X, Y = np.meshgrid(X, Y)
    Z = fv(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, color=c, alpha=alpha,
                           linewidth=0, antialiased=antialiased)


# 从-5到5的范围,绘制函数f(x, y)的向量场
# plot_vector_field(f, -5, 5, -5, 5)
plot_scalar_field(u, -5, 5, -5, 5)

# 显示调用plt.show()打开图形界面
plt.show()
