import math as ma


def length(v):
    """ 长度 """
    return ma.sqrt(v[0]**2 + v[1]**2)


def to_polar(vector):
    """ 笛卡尔坐标 -> 极地坐标 """
    x, y = vector[0], vector[1]
    angle = ma.atan2(y, x)
    return length(vector), angle


def to_cartesian(polar_vector):
    """ 极地坐标 -> 笛卡尔坐标 """
    length, angle = polar_vector[0], polar_vector[1]
    return length*ma.cos(angle), length*ma.sin(angle)


def rotate2d(angle, vector):
    """
    二维旋转函数: 函数接收一个角度和一个二维向量，并返回一个旋转的二维向量。
    :param angle:角度
    :param vector:二维向量
    :return:旋转后的二维向量
    """
    l, a = to_polar(vector)
    return to_cartesian((l, a+angle))


def rotate_z(angle, vector):
    """
    二维旋转函数(三维向量的 x 坐标和 y 坐标应用该函数):给定任意角度，柯里化版的函数产生一个做相应旋转的向量变换。
    :param angle:角度
    :param vector:三维向量
    :return:旋转后的二维向量(x,y新值,z不变)
    """
    x, y, z = vector
    new_x, new_y = rotate2d(angle, (x, y))
    return new_x, new_y, z


def rotate_z_by(angle):
    """
    三维向量变换:给定任意角度，柯里化版的函数产生一个做相应旋转的向量变换。
    :param angle:角度
    :return:三维向量
    """
    def new_function(v):
        return rotate_z(angle, v)
    return new_function


def dot(u, v):
    """
    点积函数:使用Python的zip函数对相应的坐标进行配对，然后在推导式中将每对坐标相乘，
并添加到结果列表中
    :param u: 向量,坐标数目和v相同
    :param v: 向量,坐标数目和u相同
    :return:点积值
    """
    return sum([coord1 * coord2 for coord1, coord2 in zip(u, v)])


def matrix_multiply(a, b):
    """
    点乘实现矩阵乘法:外层调用构建结果的行，内层调用构建每行的项。
    乘积矩阵的每一项是第一个矩阵的1行与第二个矩阵的1列的点积 
    :param a: 输入矩阵a已经是第一个矩阵的行元组
    :param b: zip(*b)是第二个矩阵的列元组
    :return: 相乘后的结果
    """
    return tuple( 
        tuple(dot(row, col) for col in zip(*b))
        for row in a 
    )


def infer_matrix(n, transformation):
    """
    矩阵函数: 根据线性变换和维度 打印矩阵元组
    :param n:维度参数（比如 2或3）
    :param transformation:1个线性向量变换的函数参数(必须是线性变换)
    :return:n × n方阵（一个n元组的n元组的数字集，表示线性变换的矩阵）
    """
    def standard_basis_vector(i):
        # 创建第i个标准基向量表示一个元组，在第i个坐标中包含1，在所有其他坐标中包含0
        return tuple(1 if i == j else 0 for j in range(1, n + 1))
    # 创建标准基表示n个向量的列表
    standard_basis = [standard_basis_vector(i) for i in range(1, n + 1)]
    # 将矩阵的列定义为对标准基向量进行相应线性变换的结果
    cols = [transformation(v) for v in standard_basis]
    # 按照惯例，将矩阵重构为行元组，而不是列的列表
    return tuple(zip(*cols))


def multiply_matrix_vector(matrix, vector):
    """
    向量和矩阵相乘: 一个遍历矩阵的行，另一个遍历每一行的项
    :param matrix: n * n矩阵
    :param vector: n维向量
    :return: n维向量
    """
    return tuple(
        sum(vector_entry * matrix_entry
            for vector_entry, matrix_entry in zip(row, vector))
        for row in matrix
    )


def multiply_matrix_vector2(matrix, vector):
    """
    向量和矩阵相乘: 利用输出坐标是输入矩阵行与输入向量的点积这一事实相乘
    :param matrix: n * n矩阵
    :param vector: n维向量
    :return: n维向量
    """
    return tuple(
        dot(row, vector)
        for row in matrix
    )


def matrix_power(power, matrix):
    """
    取一个矩阵的幂:对于方阵A，可以把AA写成A2，把AAA写成A3，
    :param power: 整数 >= 1
    :param matrix: 矩阵
    :return: 指定整数的矩阵的幂的结果
    """
    result = matrix
    for _ in range(1, power):
        result = matrix_multiply(result, matrix)
    return result


def transpose(matrix):
    """
    转置函数: 将列向量转换成行向量，或者将行向量转换成列向量。得到的矩阵叫作原矩阵的转置矩阵
    :param matrix:列向量
    :return:行向量
    """
    # 调用zip(*matrix)会返回矩阵中列的列表，然后再对其进行元组化,有交换任意输入矩阵中行和列的效果
    return tuple(zip(*matrix))


def rotate_2d(angle, vector):
    """
    二维矩阵旋转方法(纯代数实现): 对比 rotate2d: 无需前置length,to_polar,to_cartesian方法
    :param angle:角度
    :param vector:二维向量
    :return:旋转后的二维向量
    """
    x, y = vector
    new_x = x * ma.cos(angle) - y * ma.sin(angle)
    new_y = x * ma.sin(angle) - y * ma.cos(angle)
    return round(new_x, 2), round(new_y, 2)


def rotate_3d(angle, vector):
    """
    三维矩阵旋转方法(纯代数实现), 绕 Z轴旋转
    :param angle:角度
    :param vector:三维向量
    :return:旋转后的三维向量
    """
    x, y, z = vector
    cos_angle = ma.cos(angle)
    sin_angle = ma.sin(angle)
    new_x = x * cos_angle - y * sin_angle
    new_y = x * sin_angle - y * cos_angle
    return round(new_x, 2), round(new_y, 2), z


if __name__ == '__main__':

    # 测试向量
    a = ((1, 1, 0), (1, 0, 1), (1, -1, 1))
    b = ((0, 2, 1), (0, 1, 0), (1, 0, -1))
    c = ((1, 2), (3, 4))
    d = ((0, -1), (1, 0))
    v = (3, -2, 5)

    # 5.1部分
    print(matrix_multiply(a, b))  # 2个三维矩阵乘法
    print(matrix_multiply(c, d))  # 2个二维矩阵乘法

    # 测试矩阵函数: infer_matrix(3,rotate_z_by(pi/2))
    print(infer_matrix(3, rotate_z_by(ma.pi/2)))  # ((6.123233995736766e-17, -1.0, 0.0), (1.0, 1.2246467991473532e-16, 0.0), (0, 0, 1))

    # 测试矩阵函数: multiply_matrix_vector()
    print(multiply_matrix_vector(b, v))
    # 测试矩阵函数: multiply_matrix_vector2()
    print(multiply_matrix_vector2(b, v))

    #  5.2部分
    c = ((-1, -1, 0), (-2, 1, 2), (1, 0, -1))
    # d编码为向量(1, 1, 1),请注意，必须写成(1,)而不是(1)，以便Python将它看作一个一元组而不是一个数。
    d = ((1,), (1,), (1,))
    print(matrix_multiply(c, d))  # ((-2,), (1,), (0,))
    # 测试二维和三维向量的旋转(代数版本)
    print(rotate_2d(ma.pi / 2, (1, 0)))  # (0.0, 1.0)
    print(rotate_3d(ma.pi / 2, (1, 0, 0)))  # (0.0, 1.0, 0)











