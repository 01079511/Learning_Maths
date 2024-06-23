import matplotlib.pyplot as plt
import numpy as np

def numeric_diff(f, x):
    """ 求导函数 """

    # 定义距离h(趋于0,但不能太小,程序报错)
    h = 1e-4  # 10的-4次方
    # x-h,x,x+h
    return (f(x + h) - f(x - h))/(2*h)

def func_1(x):
    return x**2

def func_2(x):
    return x**3

def tangent_line(f, x):
    """ 切线函数 """

    # y = ax + b,a = 斜率 = 导数值
    a = numeric_diff(f, x)
    # y = f(x)
    b = f(x) - a*x

    # 定义子函数返回直线方程
    def line(t):
        return a*t + b

    return line

x = np.arange(0, 20, 0.1)
y = func_1(x)
tl = tangent_line(func_1, 10)
y2 = tl(x)  # x里面的200个点带入直线方程(发生广播操作返回200个y对应的值)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
plt.close()

d = numeric_diff(func_1, 10)
print(d)

# 第三方库(sympy)直接求导数
import sympy
from sympy import *
# x = symbols('x')
# f = x**2
# f = exp(x)  # exp高等数学里以自然常数e为底的指数函数。用途：用来表示自然常数e的指数。
# f = sin(x)
# d = sympy.diff(f)
# print(d)
# print(d.evalf(subs={'x': pi/2}))  # evalf()此函数计算给定的数值表达式，最高可达 100 位的给定浮点精度
#
# # 高阶导数
# expr = x**4
# f1 = diff(expr, x, 1)  # 1阶导
# f2 = diff(expr, x, 2)  # 2阶导
# f3 = diff(expr, x, 3)  # 3阶导
# print(f"f1:{f1},\nf2:{f2},\nf3:{f3}")
# # 带入值计算
# print(f"当x = 2 时,3阶导的值是{f3.evalf(subs={'x': 2})}")
