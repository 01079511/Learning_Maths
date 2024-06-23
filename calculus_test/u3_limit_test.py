import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = sp.Symbol('x')
f1 = sp.sin(x)/x
L = sp.limit(f1, x, 'oo')
print(f'{f1}的极限是{L}')

x1 = np.arange(-100, 100, 0.01)
y1 = np.sin(x1)/x1  # 广播操作y1->x1

# 调整输入图型大小
plt.figure(figsize=(12, 5))
# 标题
plt.title('y = sin(x)/x')

# 传入值 并show()展示
plt.plot(x1, y1)
plt.show()
plt.close()