import numpy as np
import matplotlib.pyplot as plt

# 假设data是你的数据序列
data = [10, 15, 12, 18, 25, 30, 28, 35, 40, 50]

# 拟合直线
x = np.arange(len(data))
coeffs = np.polyfit(x, data, 1)  # 一次多项式拟合
slope = coeffs[0]  # 拟合斜率

# 判断趋势
if slope > 0.1:
    trend = "上升"
elif slope < -0.1:
    trend = "下降"
else:
    trend = "平稳"

# 可视化
plt.plot(data, label='Data')
plt.plot(x, np.polyval(coeffs, x), label='Fitted Line', color='red')
plt.title(f'Trend: {trend}')
plt.legend()
plt.show()
