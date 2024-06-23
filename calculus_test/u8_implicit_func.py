from sympy import *

x, y = symbols('x y')
z = x**2 + y**2 - 4
#  idiff() 计算隐函数的导数(等式不是数值)
f = idiff(z, y, x)
print(f)
print(f.evalf(subs={'x': 2, 'y': 4}))

z2 = 5*sin(x) + 3*sec(y) - y + x**2 - 3
f2 = idiff(z2, y, x)
print(f2)
print(f2.evalf(subs={'x': 0, 'y': 0}))
