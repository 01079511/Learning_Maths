from u7_vectors import add, scale
from u9_draw2d import *
from u9_draw3d import *
from math import pi, sin, cos

# 把坐标、速度和加速度看作时间的函数
# s(t)、v(t)和 a(t)，其中s(t) = (x(t)， y(t))，v(t) = (x'(t), y'(t))，a(t) = (x''(t), y''(t))
t = 0
s = (0, 0)
v = (1, 0)
a = (0, 0.2)

# 时间常量步长Δt = 2
dt = 2
# 总共需要5个时间步
steps = 5
# 在计算过程中，可以将坐标存储在一个数组中，方便以后使用。
positions = [s]
for _ in range(0, 5):
    t += 2
    # 通过将坐标变化量∆s = v · ∆t与当前坐标s相加来更新坐标
    s = add(s, scale(dt, v))
    # 通过将速度变化量∆v = a · ∆t与当前速度v相加来更新速度
    v = add(v, scale(dt, a))
    positions.append(s)

draw2d(Points2D(*positions))


def pairs(lst):
    return list(zip(lst[:-1], lst[1:]))


def eulers_method(s0, v0, a, total_time, step_count):
    """
    创建一个函数，对一个不断加速的对象自动执行欧拉方法。
    :param s0:初始坐标向量
    :param v0:初始速度向量
    :param a:加速度向量
    :param total_time:总时间
    :param step_count:步数
    :return:
    """
    trajectory = [s0]
    s = s0
    v = v0
    dt = total_time/step_count
    for _ in range(0, step_count):
        # 通过将速度变化量∆v = a · ∆t与当前速度v相加来更新速度
        v = add(v, scale(dt, a))
        # 通过将坐标变化量∆s = v · ∆t与当前坐标s相加来更新坐标
        s = add(s, scale(dt, v))
        trajectory.append(s)
    return trajectory


approx5 = eulers_method((0, 0), (1, 0), (0, 0.2), 10, 5)
approx10 = eulers_method((0, 0), (1, 0), (0, 0.2), 10, 10)
approx100 = eulers_method((0, 0), (1, 0), (0, 0.2), 10, 100)
approx1000 = eulers_method((0, 0), (1, 0), (0, 0.2), 10, 1000)

draw2d(
    Points2D(*approx5, color='C0'),
    *[Segment2D(t, h, color='C0') for (h, t) in pairs(approx5)],
    Points2D(*approx10, color='C1'),
    *[Segment2D(t, h, color='C1') for (h, t) in pairs(approx10)],
    *[Segment2D(t, h, color='C2') for (h, t) in pairs(approx100)],
    *[Segment2D(t, h, color='C3') for (h, t) in pairs(approx1000)],
    )

"""
任何抛射物，如抛出的棒球、子弹或空中的滑雪板，都有同样的加速度向量：9.81 m/s^^2，
指向地心。如果我们把平面中的 x轴看成平地，y轴正方向指向上方，
就相当于加速度向量为（0, –9.81）。如果在x = 0处从肩高位置抛出一个棒球，
我们可以说它的初始坐标是（0, 1.5）。
假设它以30 m/s的初速度从x轴正方向向上20°的角度抛出，
用欧拉方法模拟它的轨迹。棒球沿x轴正方向大约走了多远才落地？
"""
angle = 20 * pi/180
s0 = (0, 1.5)
v0 = (30*cos(angle), 30*sin(angle))
a = (0, -9.81)

result = eulers_method(s0, v0, a, 3, 100)
# draw2d(Points2D(*result, color='C0'))


def baseball_trajectory(degrees):
    """
    重新运行上一个抛射模拟中的欧拉方法模拟，初始速度同样为 30，
    但初始坐标为(0, 0)，并尝试各种角度的初始速度。什么角度能使棒球在落地前走得最远？
    """
    radians = degrees * pi/180
    s0 = (0, 0)
    v0 = (30*cos(radians), 30*sin(radians))
    a = (0, -9.81)
    return [(x, y) for (x, y) in eulers_method(s0, v0, a, 10, 1000)
            if y >= 0]

draw2d(Points2D(*baseball_trajectory(10), color='C0'),
       Points2D(*baseball_trajectory(20), color='C1'),
       Points2D(*baseball_trajectory(45), color='C2'),
       Points2D(*baseball_trajectory(60), color='C3'))

"""
一个对象在三维空间中运动，其初速度为(1, 2, 0)，
加速度向量恒定为 (0, –1, 1)。
如果它从原点出发，10秒后会在哪里？
"""
traj3d = eulers_method((0, 0, 0), (1, 2, 0), (0, -1, 1), 10, 10)
draw3d(
    Points3D(*traj3d)
)





# 08.19

