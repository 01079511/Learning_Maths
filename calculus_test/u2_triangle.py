import numpy as np
import matplotlib.pyplot as plt

# 普林斯顿微积分读本(修订版)-U2.三角学回顾

x = np.array(np.linspace(-2*np.pi, 2*np.pi, 51))
y = x - np.cos(x) # sin cos tan cot sec csc
# y = x**2 + 1

# 调整输入图型大小
plt.figure(figsize=(5, 5))
# 调整打印范围
# plt.ylim(-1, 30)

# 传入值 并show()展示
plt.plot(x, y)
plt.show()
plt.close()