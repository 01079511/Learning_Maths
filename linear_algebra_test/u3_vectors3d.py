import math as ma
from random import random

pm1 = [1, -1]

vertices = [(x, y, z) for x in pm1 for y in pm1 for z in pm1]
edges = [((-1, y, z), (1, y, z)) for y in pm1 for z in pm1] +\
            [((x, -1, z), (x, 1, z)) for x in pm1 for z in pm1] +\
            [((x, y, -1), (x, y, 1)) for x in pm1 for y in pm1]
# print(vertices)


def to_cartesian(polar_vector):
    """ 极地坐标 -> 笛卡尔坐标 """
    length, angle = polar_vector[0], polar_vector[1]
    return length*ma.cos(angle), length*ma.sin(angle)


def random_vector_of_length(l):
    """ 根据长度和随机角度，生成笛卡尔坐标 """
    return to_cartesian((l, 2 * ma.pi * random()))


def scale(scalar, v):
    """
    标量相乘: 通过推导运算，将向量中的每个坐标乘以标量,这一个被转换成元组的生成器推导式。
    :param scalar: 标量
    :param v:向量
    :return: 将向量中的每个坐标乘以标量,结果转换成为元组输出
    """
    return tuple(scalar * coord for coord in v)


def add(*vectors):
    """
    向量加法
    :param vectors: 向量和构成的列表
    :return: 列表内各个向量和的终值
    """
    return tuple(map(sum,zip(*vectors)))


def dot(u, v):
    """
    点积函数:使用Python的zip函数对相应的坐标进行配对，然后在推导式中将每对坐标相乘，
并添加到结果列表中
    :param u: 向量,坐标数目和v相同
    :param v: 向量,坐标数目和u相同
    :return:点积值
    """
    return sum([coord1 * coord2 for coord1,coord2 in zip(u, v)])


def length(v):
    """
    向量的长度
    :param v: 坐标(2维/3维都可)
    :return: 长度
    """
    return ma.sqrt(sum([coord ** 2 for coord in v]))


def angle_between(v1,v2):
    """
    向量间角度: 反cos(角度) = u * v / |u| * |v|
    :param v1:向量,坐标数目和v2相同
    :param v2:向量,坐标数目和v1相同
    :return:向量间角度
    """
    return ma.acos(
                dot(v1, v2) /
                (length(v1) * length(v2))
            )


def cross(u, v):
    """
    求三维向量的向量积
    :param u:输入向量u
    :param v:输入向量v
    :return: u,v的向量积
    """
    ux, uy, uz = u
    vx, vy, vz = v
    return uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx


def linear_combination(scalars,*vectors):
    """
    标准基: 接收一个标量列表和相同数量的向量，并返回一个向量
    :param scalars:标量列表
    :param vectors:相同数量的向量
    :return:重构的向量
    """
    scaled = [scale(s, v) for s, v in zip(scalars, vectors)]
    return add(*scaled)


if __name__ == '__main__':
    def vectors_with_whole_number_length(max_coord=100):
        for x in range(1, max_coord):
            for y in range(1, x + 1):
                for z in range(1, y + 1):
                    if length((x, y, z)).is_integer():
                        yield (x, y, z)

    for l in list(vectors_with_whole_number_length()):
        print(l)

    # 练习3.15
    pairs = [(random_vector_of_length(3), random_vector_of_length(7))
             for i in range(0, 3)]
    for u, v in pairs:
        print("u = %s, v  = %s" % (u, v))
        print("length of u: %f, length of v: %f, dot product :%f" %
              (length(u), length(v), dot(u, v)))
    # 练习4.19
    print(linear_combination([1, 2, 3], (1, 0, 0), (0, 1, 0), (0, 0, 1)))

