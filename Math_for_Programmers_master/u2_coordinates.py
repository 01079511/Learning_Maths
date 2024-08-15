import math as ma
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def add(*vectors):
    """ 通过将所有向量各自的x坐标和y坐标相加，可以实现任意数量的向量相加 """
    return sum([v[0] for v in vectors]), sum([v[1] for v in vectors])


def length(v):
    """ 长度 """
    return ma.sqrt(v[0]**2 + v[1]**2)


def subtract(v1, v2):
    return v1[0]-v2[0], v1[1]-v2[1]


def distance(v1, v2):
    return length(subtract(v1, v2))


def perimeter(vectors):
    distances = [distance(vectors[i], vectors[(i+1) % len(vectors)])
                    for i in range(0, len(vectors))]
    return distances, sum(distances)


def scale(scalar, v):
    """ 将输入向量v和输入标量s相乘 """
    return scalar * v[0], scalar * v[1]


def translate(translation, vectors):
    """
    实现函数translate(translation, vectors)
    接收一个平移向量和一个向量列表，返回一个根据平移向量平移后的向量列表。
    例如，对于translate ((1,1), [(0,0), (0,1,), (-3,-3)]) ，它应该返回[(1,1), (1,2), (-2, -2)]。
    """
    return [add(translation, v) for v in vectors]


def to_polar(vector):
    """ 笛卡尔坐标 -> 极地坐标 """
    x, y = vector[0], vector[1]
    angle = ma.atan2(y, x)
    return length(vector), angle


def to_cartesian(polar_vector):
    """ 极地坐标 -> 笛卡尔坐标 """
    length, angle = polar_vector[0], polar_vector[1]
    return length*ma.cos(angle), length*ma.sin(angle)


def rotate(angle, vectors):
    """
    实现rotate(angle, vectors)函数
    接收笛卡儿坐标向量数组，并将这些向量旋转指定的角度（根据角度的正负来确定是逆时针还是顺时针）
    """
    polars = [to_polar(v) for v in vectors]
    return [to_cartesian((l, a+angle)) for l, a in polars]


if __name__ == '__main__':
    # 21个vector
    dino_vectors = [(6, 4), (3, 1), (1, 2)]

    # print(perimeter(dino_vectors))
    new_dino = translate((8, 8), rotate(5 * ma.pi / 3, dino_vectors))

