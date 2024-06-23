# 普林斯顿微积分读本 牛顿法的实现
def integral(f, a, b, n):
    """
    黎曼和 --python实现定积分的估算值

    参数：
    f：即f(cj)。
    a：[a, b] 区间下限
    b：[a, b] 上限
    n: n即mesh,也是子区间数量,越大越精准,但影响速度，太小影响结果，1000及以上精度高，100属一般
    具体参考 普林斯顿微积分读本第16章第2节

    返回：
    l定积分的结果
    """

    # 计算步长,如：把[0,2] 区间分成n个小区间, 第一区间是从 0 到2/n; 第二区间是从 2/n 到 4/n, 以此类推
    dx = (b - a) / n
    # 初始面积
    integral_sum = 0

    # 计算定积分
    for j in range(n):

        # c_j即定义的cj,等距划分情况 cj in [a,b]非等距划分则需要注意
        c_j = a + j * dx

        integral_sum += f(c_j) * dx

    return integral_sum

# 求解function对象
def f(x):
    return x ** 2

if __name__ == '__main__':
    result = integral(f,0, 2, 10000)
    print(f"result = {result}")

print(f"  8/3  = {8/3}")


