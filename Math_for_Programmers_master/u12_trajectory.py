import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from math import sin, cos, pi


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
        x = vx * dt
        z = vz * dt
        ts.append(t)
        xs.append(x)
        zs.append(z)
    return ts, xs, zs


def plot_trajectories(*trajs, show_seconds=False):
    """
    plot_trajectories 函数，它将 trajectory 函数的输出结果作为输入，
    并将其传给Matplotlib的plot函数，绘制曲线来显示每个炮弹的飞行轨迹
    :param trajs:trajectory 函数的输出结果
    :param show_seconds:
    :return:
    """
    for traj in trajs:
        xs, zs = traj[1], traj[2]
        plt.plot(xs, zs)
        if show_seconds:
            second_indices = []
            second = 0
            for i, t in enumerate(traj[0]):
                if t >= second:
                    second_indices.append(i)
                    second += 1
            plt.scatter([xs[i] for i in second_indices], [zs[i] for i in second_indices])
    xl = plt.xlim()
    plt.plot(plt.xlim(), [0, 0], c='k')
    plt.xlim(*xl)

    width = 7
    coords_height = (plt.ylim()[1] - plt.ylim()[0])
    coords_width = (plt.xlim()[1] - plt.xlim()[0])
    plt.gcf().set_size_inches(width, width * coords_height / coords_width)


plot_trajectories(trajectory(45), trajectory(60))
plt.show()

