from math import sqrt, sin, cos, acos, atan2

# def add(v1,v2):
#     return (v1[0] + v2[0], v1[1] + v2[1])

# def add(*vectors):
#     by_coordinate = zip(*vectors)
#     coordinate_sums = [sum(coords) for coords in by_coordinate]
#     return tuple(coordinate_sums)


def add(*vectors):
    """
    向量求和: 对多个向量进行逐位求和
    zip(*vectors): 将输入向量根据坐标分组,如:((1, 2), (3, 4), (5, 6)) ==> [(1, 3, 5), (2, 4, 6)]
    map(sum, zip(*vectors)): 对第二次参数，即目标，如 每个坐标组，调用第一函数，如 sum
    :param vectors: 任意数量的向量组,如((1, 2), (3, 4), (5, 6))
    :return: 求和结果，如上述参数返回: (9, 12)
    """
    return tuple(map(sum, zip(*vectors)))


def subtract(v1, v2):
    """
    向量减法
    :param v1: 第一个向量
    :param v2: 第二个向量
    :return: 返回向量差值
    """
    return tuple(v1-v2 for (v1, v2) in zip(v1, v2))


def length(v):
    """
    向量的长度 / 计算向量v的欧几里得长度
    实现了向量从原点到向量坐标的距离计算
    如:v = (3, 4) 从 [coord ** 2 for coord in v] => [9, 16] 再 sum[9, 16] => 25,最后 sqrt(25) => 5
    :param v: 坐标(2维/3维都可)
    :return: 长度
    """
    return sqrt(sum([coord ** 2 for coord in v]))


def dot(u, v):
    """
    点积函数:计算u,v间的点积
    zip(*vectors): 将输入向量根据坐标分组,如:((1, 2, 3), (4, 5, 6)) ==> [(1, 4), (2, 5), (3, 6)]
    [coord1 * coord2 for coord1, coord2 in zip(u, v)],如上,输入后得到[1*4, 2*5, 3*6] => [4, 10, 18]
    最后 sum([上述结果]) => sum([4, 10, 18]) => 4+10+18 => 32
    :param u: 向量,坐标数目和v相同
    :param v: 向量,坐标数目和u相同
    :return:点积值
    """
    return sum([coord1 * coord2 for coord1, coord2 in zip(u, v)])


def distance(v1, v2):
    """
    2个向量的距离
    subtract(v1, v2) => v1-v2
    length(v1-v2) => 2个向量插值的向量的长度
    :param v1:
    :param v2:
    :return: 2个向量间欧几里得距离
    """
    return length(subtract(v1, v2))


def perimeter(vectors):
    """
    计算一组向量表示的多边形的周长
    %len(vectors) 是取模运算，确保索引在0->len(vectors)之间,表示从最后一个顶点指向第一个顶点
    如 vectors 4个顶点，0,1,2,3
    i = 0,(i+1)%len(vectors) = 1,指向下一个顶点1
    i = 1,(i+1)%len(vectors) = 2,指向下一个顶点2
    i = 2,(i+1)%len(vectors) = 3,指向下一个顶点3
    i = 3,(i+1)%len(vectors) = 0,指向下一个顶点0
    distance(2个顶点)=>2个顶点距离，循环得到一个包含所有距离的[...],对其sum求和,得到周长
    :param vectors: 一组向量
    :return: 多边形的周长
    """
    distances = [distance(vectors[i], vectors[(i+1)%len(vectors)])
                    for i in range(0, len(vectors))]
    return sum(distances)


def scale(scalar, v):
    """
    标量相乘: 将向量v按照标量scalar进行缩放
    :param scalar: 标量
    :param v:向量
    :return: 缩放后的元组
    """
    return tuple(scalar * coord for coord in v)


def to_cartesian(polar_vector):
    """
    极地坐标 -> 笛卡尔坐标
    解包: 提取polar_vector极地坐标的长度和角度
    转换: 使用三角函数 极地坐标 ==> 笛卡尔坐标
    """
    length, angle = polar_vector[0], polar_vector[1]
    return length*cos(angle), length*sin(angle)


def rotate2d(angle, vector):
    """
    将二维向量vector旋转一个角度angle
    to_polar(vector) 实现 向量 -> 极地坐标
    将传入的角度和原角度相加，再从 极地坐标 -> 笛卡尔坐标
    """
    l, a = to_polar(vector)
    return to_cartesian((l, a+angle))


def translate(translation, vectors):
    """
    将一组向量vectors,平移一个向量translation,即
    列表生成式子: 对每个向量v，计算 translation + v
    :param translation:
    :param vectors:
    :return: 平移后的向量组
    """
    return [add(translation, v) for v in vectors]


def to_polar(vector):
    """
    笛卡尔坐标 -> 极地坐标
    解包: 提取向量的x,y坐标
    计算角度: atan2(y, x)
    :param vector:
    :return: 极地坐标
    """
    x, y = vector[0], vector[1]
    angle = atan2(y, x)
    return length(vector), angle


def angle_between(v1, v2):
    """
    计算2个向量之间的夹角
    2个向量间点积: dot(v1, v2)
    2个向量间长度的乘积: length(v1) * length(v2)
    计算点积与长度乘积执笔的反余弦 得到夹角
    :param v1:
    :param v2:
    :return:夹角
    """
    return acos(
                dot(v1, v2) /
                (length(v1) * length(v2))
            )


def cross(u, v):
    """
    计算2个三维向量的叉积
    :param u:
    :param v:
    :return: 根据公式 叉积 = uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx
    """
    ux, uy, uz = u
    vx, vy, vz = v
    return uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx


def component(v, direction):
    """
    计算向量v在方向direction上的分量
    :param v:
    :param direction:
    :return: 分量 = 点积/方向长度
    """
    return dot(v, direction) / length(direction)


def unit(v):
    """
    将向量v归一化，得到单位向量
    :param v:
    :return: 根据 1./length(v)计算向量到1的缩放比例，通过scale按比例缩放得到自己的单位向量
    """
    return scale(1./length(v), v)


def linear_combination(scalars, *vectors):
    """
    计算一组向量的线性组合
    zip(scalars, vectors): 将标量和向量配对,如  (2,(1,2)) (3,(3, 4))
    列表生成式: 对配对后的结果中的每个组合，进行缩放得到,如 (2*1,2*2=4) (3*3=9,3*4=12)
    add(*scaled)对缩放后的结果进行向量相加,add(*[(2, 4), (9, 12)]) => (11, 16)
    :param scalars: 标量组,如 [s1 = 2,s2 = 3]
    :param vectors: 向量组,如: v1 = (1, 2),v2 = (3, 4)
    :return: 如: s1 * v1 + s2 *v2
    """
    scaled = [scale(s, v) for s, v in zip(scalars, vectors)]
    return add(*scaled)


def plane_equation(p1, p2, p3):
    """
    实现类: 输入是三个三维点,返回它们所在平面的标准方程;
    示例: 如果标准方程是ax + by + cz = d，则函数可以返回元组(a, b, c, d)
    逻辑: 如果给定的点是p1、p2、p3,
    那么向量差p3 − p1和p2 − p1平行于平面。那么向量积(p2 − p1) × (p3 − p1)就垂直于平面,
    (只要p1、p2、p3三点组成一个三角形，向量差之间就不平行),有了平面上的一点(如p1)和一个垂直的向量,
    根据两个垂直的向量点积为0,得到ax+by=c,垂线焦点(x0,y0),(a,b)*(x0,y0) = c,
    即,两个垂直的向量点积为d,(a, b, c) · p1 = d 得出d
    :param p1:三维点之一
    :param p2:三维点之一
    :param p3:三维点之一
    :return:标准方程的参数元组
    """
    parallel1 = subtract(p2, p1)
    parallel2 = subtract(p3, p1)
    a, b, c = cross(parallel1, parallel2)
    d = dot((a, b, c), p1)
    return a, b, c, d

# 测试plane_equation()
# print(plane_equation((1,1,1), (3,0,0), (0,3,0)))


