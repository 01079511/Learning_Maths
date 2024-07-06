# 普林斯顿微积分读本 牛顿法的实现
def newton(x0, epsilon=1e-10, max_iter=100):
    """
    使用牛顿法近似求解 f(x) = 0 的解。

    参数：
    x0：初始猜测的解。
    epsilon：停止条件，满足 |f(x)| < epsilon, 其中|f(x)| = b,就是牛顿公式里的b,而< epsilon就是比一个很小的数更小，就是趋近于0,即公式里的含义(详情见普林斯顿微积分读本260页)
    max_iter：最大迭代次数。

    返回：
    xn:近似解。
    n :迭代次数
    """

    xn = x0
    n = 0

    def Df(x):
        # 计算 f(x) 的导数公式 Df,h趋近于0的时候,如 0.00001
        h = 1e-10
        return (f(x + h) - f(x)) / h

    for r in range(max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            return xn, n
        Dfxn = Df(xn)
        if Dfxn == 0:
            return None
        xn = xn - fxn / Dfxn
        n += 1
    return None


# 测试牛顿法求解 x^5 + 2x - 1 = 0 的近似解
def f(x):
    return x**5 + 2*x - 1


approx_solution, n = newton(0)

if __name__ == '__main__':
    if approx_solution is not None:
        print(f"找到解 = {approx_solution:.4f},经过 {n} 次迭代." )
    else:
        # 牛顿法则有不使用的情况，按异常不做处理抛出即可
        print("未找到解！")