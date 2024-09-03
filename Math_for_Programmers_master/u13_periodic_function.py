import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""
结合第13章方向,实现个人需求:
1.特征提取:
从傅里叶变换的结果中，提取信号的频谱，选择2个重要频率成分(频率和对应的幅度)作为特征和目标值
2.创建数据集：
将频率作为特征x,将幅度作为目标值y(虽然是一个简单示例，但是突出如何将频率特征提取出来，并用于建模)
3.训练线性回归
将频率作为特征输入，使用线性回归模型来预测对应的幅度值。模型的性能通过均方差(MSE)评估
4.预测与评估
调用训练好的模型对测试数据进行预测，并通过绘图展示结果
"""

# 1. 生成时间序列数据
N = 500  # 数据点数量
t = np.linspace(0, 10, N)  # 500个点的时间序列t
# 信号包含两个频率成分(正弦波频率1Hz,振幅0.5,另一正弦波频率3Hz,振幅0.2)，并添加噪声
signal = 0.5 * np.sin(2 * np.pi * 1 * t) + 0.2 * np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(N)

# 2. 进行傅立叶变换并提取特征,np.fft.fft()函数对时间序列数据进行快速傅里叶变换(FFT),将数据从时间域转换到频率域
signal_fft = np.fft.fft(signal)
freqs = np.fft.fftfreq(N, t[1] - t[0])  # freqs是频率轴对应于傅里叶变换结果中的频率成分

# 只使用前半部分频谱（因为傅立叶变换结果是对称的）
amplitude_spectrum = np.abs(signal_fft[:N // 2])  # 显示不同频率下的信号强度
important_freqs = freqs[:N // 2]

# 取出最重要的频率特征（比如前两个主要频率成分）
top_n = 2
top_indices = np.argsort(amplitude_spectrum)[top_n:]   # 提取前2个频率
top_freqs = important_freqs[top_indices]
top_amplitudes = amplitude_spectrum[top_indices]

# 3. 创建数据集
# 我们假设这两个频率成分的幅度是目标值y（简单示例）
X = top_freqs.reshape(-1, 1)  # 将频率作为特征
y = top_amplitudes  # 幅度作为目标值

# 4. 训练简单的线性回归模型(test_size=0.5 ,一半模拟一般测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
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
