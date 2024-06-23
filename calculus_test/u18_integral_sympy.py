import sympy as sp
import math as m

# 定义x
x = sp.Symbol('x')

# 定义函数
f1 = sp.E**2 + 2*x
f2 = x**2*sp.cos(x**3)

# 定义上限
up = (m.pi/2)**(1/3)

if __name__ == '__main__':
    # 不定积分
    result = sp.integrate(f2, x)
    print(result)

    # 定积分
    result1 = sp.integrate(f2, (x, 0, up))
    print(result1)