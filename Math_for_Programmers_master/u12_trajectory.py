import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from math import sin, cos, pi, sqrt
from u7_vectors import to_polar, to_cartesian, length


def trajectory(theta, speed=20, height=0, dt=0.01, g=-9.81):
    """
    目的:
    用欧拉方法模拟运动,建立一个模拟器来计算炮弹的飞行轨迹,
    接收发射角度以及其他一些控制参数，并返回炮弹随时间变化的位置，直到炮弹掉落在地球上;
    速度 v 被定义为其速度向量的大小，即v = |v|,则炮弹速度的x(水平方向)和z(垂直方向)分量
    分别为vx = |v| · cos(θ)和 vz = |v| · sin(θ);
    :param theta:角度(度)
    :param speed:速度
    :param height:高度
    :param dt:时间增量
    :param g:重力场强度/重力加速度
    :return:
    """
    vx = speed * cos(pi * theta / 180)  # 计算初始速度的x和z分量，将输入角度的单位从度转换为弧度
    vz = speed * sin(pi * theta / 180)
    t, x, z = 0, 0, height
    ts, xs, zs = [t], [x], [x]  # 初始化在模拟过程中保存的所有时间值和x、z位置的列表
    while z >= 0:  # 仅当炮弹在地面之上时运行模拟器
        t += dt  # 更新时间、速度 z 和位置。没有力作用在 x方向上，所以x速度不变
        vz += g * dt  # Δv = a · Δt = g (x, y) · Δt
        x += vx * dt
        z += vz * dt
        ts.append(t)
        xs.append(x)
        zs.append(z)
    return ts, xs, zs


def plot_trajectories(*trajs, show_seconds=False):
    """
    plot_trajectories 函数，它将 trajectory 函数的输出结果作为输入，
    并将其传给Matplotlib的plot函数，绘制曲线来显示每个炮弹的飞行轨迹
    :param trajs: *trajs表示可变参数,每个结果都包含 trajectory 函数的输出结果(时间,坐标x,坐标z)
    :param show_seconds: 可选参数,如果=True,将标记每 1 秒钟的位置点
    :return:
    """
    # 循环遍历绘制每条轨迹
    for traj in trajs:
        xs, zs = traj[1], traj[2]
        plt.plot(xs, zs)  # 绘制轨迹曲线
        #显示每秒的轨迹点
        if show_seconds:
            second_indices = []  # 初始化空列表,存储每秒时刻的索引
            second = 0  # 初始化变量
            for i, t in enumerate(traj[0]):  # 遍历ts序列时间t以及其索引i
                if t >= second:  # 如果当前时间t超过或等于second,表示已经超过或达到下一整秒
                    second_indices.append(i)  # 将当前时间t的索引i追加到second_indicies中
                    second += 1  # 递增1,用于查找下一整秒的位置
            plt.scatter([xs[i] for i in second_indices], [zs[i] for i in second_indices])  # matplotlib的scatter()函数在轨迹曲线上标出每秒的点
    # 绘制地平线
    xl = plt.xlim()  # 保存当前x轴的限界,以便稍后恢复
    plt.plot(plt.xlim(), [0, 0], c='k')  # 绘制y=0的黑色水平线,plt.plot()将绘制一条从xmin到xmax的水平线
    plt.xlim(*xl)  # 恢复x州的范围到原来的限界xl,确保地面线的绘制不会改变原本的x轴范围
    # 调整图形尺寸比例
    width = 7
    coords_height = (plt.ylim()[1] - plt.ylim()[0])  # 计算y轴范围的差集,即图形在y轴方向的尺寸
    coords_width = (plt.xlim()[1] - plt.xlim()[0])  # 计算x轴范围的差集,即图形在x轴方向的尺寸
    plt.gcf().set_size_inches(width, width * coords_height / coords_width)  # 调整图形的尺寸,让图形的宽高与坐标轴的宽高匹配,避免图形失真


def landing_position(traj):
    """
    测量弹道的属性: 炮弹的射程
    :param traj:trajectory 函数的输出(带有时间和x、z位置信息的列表)
    :return:对于输入的轨迹traj，traj[1]列出了x坐标，而traj[1][−1]是列表中的最后一个条目
    """
    return traj[1][-1]


def hang_time(traj):
    """
    测量弹道的属性: 滞空时间（炮弹在空中停留的时间）
    :param traj:trajectory 函数的输出(带有时间和x、z位置信息的列表)
    :return:
    """
    return traj[0][-1]


def max_height(traj):
    """
    测量弹道的属性: 飞行的最大高度
    :param traj:trajectory 函数的输出(带有时间和x、z位置信息的列表)
    :return:最大高度是z位置的最大值，即弹道输出的第三个列表中的最大值
    """
    return max(traj[2])


# 测试发射角对落地位置影响的一种方法是对几个不同的θ（theta）值计算组合函数landing_position  (trajectory(theta))的结果，并将其传递给 Matplotlib 的 scatter 函数
angles = range(0, 90, 5)  # 发射角取range(0, 90, 5),即从0到90°的角度,以5°为增量
landing_positions = [landing_position(trajectory(theta))
                     for theta in angles]
# plt.scatter(angles, landing_positions)  # 纵坐标是落地远近,横坐标是角度,显示峰值在45度


def plot_trajectory_metric(metric, thetas, **settings):
    """
    在给定的一组 θ值集合上绘制想要的任何度量结果
    :param metric: 落地位置
    :param thetas: 发射角
    :param settings:
    :return:
    """
    plt.scatter(thetas, [metric(trajectory(theta, **settings)) for theta in thetas])


def plot_function(f, xmin, xmax, **kwargs):
    """
    计算最佳射程: 绘制位置和角度
    """
    ts = np.linspace(xmin, xmax, 1000)
    plt.plot(ts, [f(t) for t in ts], **kwargs)


# 通过z(t)和r(theta)计算最佳射程(待理解)
def z(t):
    """
    要得到位置函数z(t)，需要对加速度z''(t)进行两次积分。第一次积分可以得到速度,第二次积分可以得到位置。
    根据模拟中的数据，初始速度为20 m/s，发射角为45°
    z'(t) = |v| * sin(θ) + g*t
    z''(t) = |v| * sin(θ) * t + (g/2)*t**2
    """
    return 20*sin(45*pi/180)*t + (-9.81/2)*t**2


def r(theta):
    """
    r = vx · Δt = |v|cos(θ) · Δt，所以射程 r关于发射角θ的完整函数表达式为:
    r(θ) = -(2*|v|**2)/g * sin(θ) * cos(θ)
    """
    return (-2*20*20/-9.81)*sin(theta*pi/180)*cos(theta*pi/180)


# trajectory3d前置地形函数
def flat_ground(x, y):
    """
    flat_ground() 表示平坦的地面，其中每个(x, y)点的海拔高度都为零
    模拟平面z = 0上方或下方的海拔高度，它为每一个(x, y)点返回一个数
    :param x:
    :param y:
    :return:
    """
    return 0


def ridge(x, y):
    """
    ridge() 表示两个山谷之间的山脊
    在这个山脊上，地面从原点开始，在x轴正负方向同时向上倾斜，在y轴正负方向同时向下倾斜。
    :param x:
    :param y:
    :return:
    """
    return (x**2 - 5*y**2) / 2500


def trajectory3d(theta, phi, speed=20,
                 height=0, dt=0.01, g=-9.81,
                 elevation= flat_ground, drag=0):
    """
    3维弹道函数
    之前初始速度的x分量是vx = |v|cos(θ),
    现在添加cos(ϕ)因子来表示vx = |v|cos(θ)cos(ϕ)。初始速度的y分量是vy = |v|cos(θ)sin(ϕ)
    加入阻力: Fd = −αv,因为炮弹的质量不变，所以可以使用一个阻力常数，即α/m。
    阻力引起的加速度分量为vxα/m、vyα/m和vzα/m
    :param theta: 发射角θ
    :param phi: ϕ,炮弹从x轴正方向横向旋转的角度
    :param speed:
    :param height:
    :param dt:
    :param g:
    :return:
    """
    vx = 20 * cos(pi * theta / 180) * cos(pi * phi / 180)
    vy = 20 * cos(pi * theta / 180) * sin(pi * phi / 180)  # 计算初始y速度
    vz = 20 * sin(pi * theta / 180)
    t, x, y, z = 0, 0, 0, height
    ts, xs, ys, zs = [t], [x], [y], [z]  # 存储在整个模拟过程中的时间值以及x、y和z的位置
    while z >= elevation(x, y):  # 关键字参数elevation来定义地形(默认为平地)
        t += dt
        vx -= (drag * vx) * dt  # 根据阻力的比例减小vx和vy
        vy -= (drag * vy) * dt
        vz += (g - (drag * vz)) * dt  # 通过重力和阻力的作用改变z速度（vz）
        x += vx * dt
        y += vy * dt  # 在每次迭代中更新y的位置
        z += vz * dt
        ts.append(t)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return ts, xs, ys, zs


def plot_trajectories_3d(*trajs, elevation=flat_ground,
                         bounds=None, zbounds=None, shadows=False):
    """
    实现 plot_trajectories_3d 函数，绘制 trajectory3D 的结果以及指定的地形
    :param *trajs: *trajs表示可变参数,每个结果都包含 trajectory3D 函数的输出结果
    :param elevation: 指定的地形,默认平地面
    :param bounds:
    :param zbounds:
    :param shadows:
    :return:
    """
    fig, ax = plt.gcf(), plt.figure().add_subplot(111, projection='3d')
    fig.set_size_inches(7, 7)

    if not bounds:
        xmin = min([x for traj in trajs for x in traj[1]])
        xmax = max([x for traj in trajs for x in traj[1]])
        ymin = min([x for traj in trajs for x in traj[2]])
        ymax = max([x for traj in trajs for x in traj[2]])

        padding_x = 0.1 * (xmax - xmin)
        padding_y = 0.1 * (ymax - ymin)
        xmin -= padding_x
        xmax += padding_x
        ymin -= padding_y
        ymax += padding_x

    else:
        xmin, xmax, ymin, ymax = bounds

    plt.plot([xmin, xmax], [0, 0], [0, 0], c='k')
    plt.plot([0, 0], [ymin, ymax], [0, 0], c='k')

    g = np.vectorize(elevation)
    ground_x = np.linspace(xmin, xmax, 20)
    ground_y = np.linspace(ymin, ymax, 20)
    ground_x, ground_y = np.meshgrid(ground_x, ground_y)
    ground_z = g(ground_x, ground_y)
    ax.plot_surface(ground_x, ground_y, ground_z, cmap=cm.coolwarm, alpha=0.5,
                    linewidth=0, antialiased=True)
    for traj in trajs:
        ax.plot(traj[1], traj[2], traj[3])
        if shadows:
            ax.plot([traj[1][0], traj[1][-1]],
                    [traj[2][0], traj[2][-1]],
                    [0, 0],
                    c='gray', linestyle='dashed')

    if zbounds:
        ax.set_zlim(*zbounds)


# 12.3.3 在三维空间中求炮弹的射程 P384
B = 0.001  # <1>山脊形状、发射速度和重力加速度常数
C = 0.005
v = 20
g = -9.81


def velocity_components(v, theta, phi):  # <2>一个辅助函数，用于求初始速度的x、y和z分量
    vx = v * cos(theta * pi / 180) * cos(phi * pi / 180)
    vy = v * cos(theta * pi / 180) * sin(phi * pi / 180)
    vz = v * sin(theta * pi / 180)
    return vx, vy, vz


def landing_distance(theta, phi):  # <3>初始速度的水平分量(平行于xy平面)
    vx, vy, vz = velocity_components(v, theta, phi)
    v_xy = sqrt(vx ** 2 + vy ** 2)  # <4>初始速度的水平分量(平行于xy平面)

    a = (g / 2) - B * vx ** 2 + C * vy ** 2  # <5>常数a和b
    b = vz
    landing_time = -b / a  # <6>求解落地时间的二次方程，即−b/a
    landing_distance = v_xy * landing_time  # <7>水平距离
    return landing_distance


# 绘制射程与发射参数的关系图
def scalar_field_heatmap(f, xmin, xmax, ymin, ymax, xsteps=100, ysteps=100):
    fig = plt.figure()
    fig.set_size_inches(7, 7)

    fv = np.vectorize(f)

    X = np.linspace(xmin, xmax, xsteps)
    Y = np.linspace(ymin, ymax, ysteps)
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


# 射程函数的梯度
def plot_scalar_field(f, xmin, xmax, ymin, ymax, xsteps=100, ysteps=100, c=None,
                      cmap=cm.coolwarm, alpha=1, antialiased=False):
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    ax = fig.add_subplot(111, projection='3d')

    fv = np.vectorize(f)

    # Make data.
    X = np.linspace(xmin, xmax, xsteps)
    Y = np.linspace(ymin, ymax, ysteps)
    X, Y = np.meshgrid(X, Y)
    Z = fv(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, color=c, alpha=alpha,
                           linewidth=0, antialiased=antialiased)


# 利用梯度寻找上坡方向,第一步是要能够近似地计算出任意一点的梯度。为此，可使用第9章中介绍的方法：取短割线的斜率。
def secant_slope(f, xmin, xmax):
    """
    求x值在xmin和xmax之间的割线f(x)的斜率
    :param f:
    :param xmin:
    :param xmax:
    :return:
    """
    return (f(xmax) - f(xmin)) / (xmax - xmin)


def approx_derivative(f, x, dx=1e-6):
    """
    近似导数是x − 10**(−6)和x + 10**(−6)之间的一条割线
    :param f:
    :param x:
    :param dx: 容差(tolerance)
    :return:
    """
    return secant_slope(f, x-dx, x+dx)


def approx_gradient(f, x0, y0, dx=1e-6):
    """
    为了求函数f(x, y)在点(x0, y0)处的近似偏导数，可以固定x = x0，求相对于y的导数；或者固定y = y0，求相对于x的导数。
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
    return partial_x, partial_y


def landing_distance_gradient(theta, phi):
    """
    函数r(θ, ϕ)被实现为landing_distance 函数。
    实现一个特殊函数approx_gradient，表示它的梯度,
    定义了一个向量场：由空间中每个点上的向量构成。在这种情况下，向量场表明了r在任意点(θ, ϕ)处的最大增量向量
    :param theta: θ
    :param phi: ϕ
    :return:梯度
    """
    return approx_gradient(landing_distance, theta, phi)


# 实现显示在r(θ, ϕ)热图上绘制的landing_distance_gradient的2个前置功能
def draw_arrow(tip, tail, color='k'):
    tip_length = (plt.xlim()[1] - plt.xlim()[0]) / 20.
    length = sqrt((tip[1] - tail[1]) ** 2 + (tip[0] - tail[0]) ** 2)
    new_length = length - tip_length
    new_y = (tip[1] - tail[1]) * (new_length / length)
    new_x = (tip[0] - tail[0]) * (new_length / length)
    plt.gca().arrow(tail[0], tail[1], new_x, new_y,
                    head_width=tip_length / 1.5,
                    head_length=tip_length / 2,
                    fc=color, ec=color)


def plot_vector_field(f, xmin, xmax, ymin, ymax, xsteps=10, ysteps=10, color='k'):
    X, Y = np.meshgrid(np.linspace(xmin, xmax, xsteps), np.linspace(ymin, ymax, ysteps))
    U = np.vectorize(lambda x, y: f(x, y)[0])(X, Y)
    V = np.vectorize(lambda x, y: f(x, y)[1])(X, Y)
    plt.quiver(X, Y, U, V, color=color)
    fig = plt.gcf()


# 实现梯度上升
def gradient_ascent(f, xstart, ystart, tolerance=1e-6):
    """
,   梯度上升算法的输入是需要最大化的函数和一个起点位置,最终，当接近一个最大值时，梯度会随着图形到达一个高点而接近零
    :param f:
    :param xstart: 将(x, y)的初始值设置为输入值
    :param ystart: 将(x, y)的初始值设置为输入值
    :param tolerance: 容差(tolerance)当梯度接近零时，再也没有上坡路可走了，算法就此终止。通过容差，表示应该遵循的最小梯度值。如果梯度小于容差，就可以确保图形是平的，已经达到了函数的最大值
    :return:
    """
    x = xstart
    y = ystart
    grad = approx_gradient(f, x, y)  # 告诉我们如何从当前(x, y)点处上坡
    while length(grad) > tolerance:  # 仅当梯度大于容差时，才前进至新点
        x += grad[0]  # 将(x, y)更新为(x, y) + ∇f(x, y）
        y += grad[1]
        grad = approx_gradient(f, x, y)  # 更新新点的梯度
    return x, y  # 当无上坡路可走时,返回x和y的值


def gradient_ascent_points(f, xstart, ystart, tolerance=1e-6):
    """
    了更好地了解 gradient_ascent()算法的工作原理，在θϕ平面上跟踪梯度上升的轨迹。
    这与通过欧拉方法迭代跟踪时间和位置值的方式类似。
    :param f:
    :param xstart:
    :param ystart:
    :param tolerance:
    :return:
    """
    x = xstart
    y = ystart
    xs, ys = [x], [y]
    grad = approx_gradient(f,x,y)
    while length(grad) > tolerance:
        x += grad[0]
        y += grad[1]
        grad = approx_gradient(f,x,y)
        xs.append(x)
        ys.append(y)
    return xs, ys


# 各类测试
# 显示发射角为45°与60°的发射曲线
# plot_trajectories(trajectory(45), trajectory(60))
# show_seconds=True每过1秒就在轨迹图上画一个大圆点，从而在图上体现时间的流逝。
# plot_trajectories(trajectory(20), trajectory(45), trajectory(60), trajectory(80), show_seconds=True)
# 绘制当发射角为10°、20°和30°时落地位置与发射角关系的散点图,模拟的初始发射高度设为10米
# plot_trajectory_metric(landing_position, [10, 20, 30], height=10)
# 如果炮弹的初始发射高度为10米，那么它达到最大射程的近似发射角度是多少？
# plot_trajectory_metric(landing_position, range(0, 90, 5), height=10)
# z的二阶导数图像
# plot_function(z, 0, 2.9)  # 纵坐标速度,横坐标时间
# 对应角度在模拟中得到的落地位置绘制出来
# plot_function(r,  0, 90)  # # 纵坐标落地位置,横坐标角度
# 当炮弹向下坡方向发射时落在z = 0以下，向上坡方向发射时落在z = 0以上
# plot_trajectories_3d(
#     trajectory3d(20, 0, elevation=ridge),
#     trajectory3d(20, 270, elevation=ridge),
#     bounds=[0, 40, -40, 0],
#     elevation=ridge)
# 添加阻力系数drag=0.1后的效果
# plot_trajectories_3d(
#     trajectory3d(20, -20, elevation=ridge),
#     trajectory3d(20, -20, elevation=ridge, drag=0.1),
#     bounds=[0, 40, -40, 0],
#     elevation=ridge)
# 绘制射程与发射参数的关系图
# scalar_field_heatmap(landing_distance, 0, 90, 0, 360)
# 射程函数的梯度图
# plot_scalar_field(landing_distance,0,90,0,360)
# 热图上绘制的landing_distance_gradient,在函数r(θ, ϕ)的热图上绘制梯度向量场∇r(θ, ϕ)。箭头指向r增加的方向，也就是热图上较亮的点
# scalar_field_heatmap(landing_distance,0, 90, 0, 360)
# plot_vector_field(landing_distance_gradient, 0, 90, 0, 360, xsteps=10, ysteps=10, color='k')
# plt.xlabel('theta')
# plt.ylabel('phi')
# plt.gcf().set_size_inches(9, 7)
# 通过更改输入方程(确保无递归),发现 plot_vector_field 功能不在问题,问题在原函数f中,landing_distance_gradient()书中内容有误
# def f(x, y):
#     return -2 * y, x
# plot_vector_field(f, -5, 5,-5,5)
# 在(θ, ϕ) = (37.5°, 90°)附近相同的图，这是其中一个最大值的近似位置
# scalar_field_heatmap(landing_distance,35,40,80,100)
# plot_vector_field(landing_distance_gradient,35,40,80,100,xsteps=10,ysteps=15,color='k')
# plt.xlabel('theta')
# plt.ylabel('phi')
# plt.gcf().set_size_inches(9,7)
# 在(θ, ϕ) = (36°, 83°)处测试一下gradient_ascent()功能
# print(gradient_ascent(landing_distance, 36, 83))
# 绘制梯度上升算法求出的射程函数最大值的路径,通过gradient_ascent_points()作为输入函数
# scalar_field_heatmap(landing_distance,35,40,80,100)
# plt.scatter([36,37.58114751557887],[83,89.99992616039857],c='k',s=75)
# plt.plot(*gradient_ascent_points(landing_distance,36,83),c='k')
# plt.xlabel('theta')
# plt.ylabel('phi')
# plt.gcf().set_size_inches(9,7)
# 发射角(37.58°, 90°)和(37.58°, 270°)都能使函数 r(θ, ϕ)达到最大值，因此，这两个发射角都能使炮弹产生最大射程。该射程约为53米。
# print(landing_distance(37.58114751557887, 89.99992616039857))

# 显式调用图形窗口
plt.show()


