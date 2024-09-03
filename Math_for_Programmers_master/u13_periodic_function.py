import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 生成时间序列数据
N = 500
t = np.linspace(0, 10, N)
# 信号包含两个频率成分，并添加噪声
signal = 0.5 * np.sin(2 * np.pi * 1 * t) + 0.2 * np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(N)

# 2. 进行傅立叶变换并提取特征
signal_fft = np.fft.fft(signal)
freqs = np.fft.fftfreq(N, t[1] - t[0])

# 只使用前半部分频谱（因为傅立叶变换结果是对称的）
amplitude_spectrum = np.abs(signal_fft[:N // 2])
important_freqs = freqs[:N // 2]

# 取出最重要的频率特征（比如前两个主要频率成分）
top_indices = np.argsort(amplitude_spectrum)[-2:]
top_freqs = important_freqs[top_indices]
top_amplitudes = amplitude_spectrum[top_indices]

# 3. 创建数据集
# 我们假设这两个频率成分的幅度是目标值y（简单示例）
X = top_freqs.reshape(-1, 1)  # 将频率作为特征
y = top_amplitudes  # 幅度作为目标值

# 4. 训练简单的线性回归模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 进行预测并评估模型
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 绘制回归直线与测试数据
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Test Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Linear Regression on Frequency-Amplitude Data')
plt.legend()
plt.show()