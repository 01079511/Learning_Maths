import numpy as np
import matplotlib.pyplot as plt

# 普林斯顿微积分读本(修订版)-U1.函数 图像和直线
# x = np.array(np.arange(-5, 5, 0.2))
# y = x**2

x = np.array(np.arange(-1, 1, 0.00001))
# y = np.sin(x)/x
y = np.cos(x)

# 50个点或元素 连续步长0.2
print(len(x))

# 调整输入图型大小
plt.figure(figsize=(5, 5))
# 调整打印范围
plt.ylim(-1, 2)

# 传入值 并show()展示
plt.plot(x, y)
plt.show()
plt.close()
