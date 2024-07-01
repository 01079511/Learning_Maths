import numpy as np
from v7_vectors import distance


def standard_form(v1, v2):
    """
    将一条直线的2个点v1, v2 转换为直线的标准形式:
    ax + by = c,
    a = (y2 - y1), b = -(x2 − x1) = (x1 − x2), c = (x1y2 - x2y1)
    :param v1:
    :param v2:
    :return:直线的标准形式参数
    """
    x1, y1 = v1
    x2, y2 = v2
    a = y2 - y1
    b = x1 - x2
    c = x1 * y2 - y1 * x2
    return a, b, c


def intersection(u1, u2, v1, v2):
    """
    计算2条直线的交点
    利用standard_form函数将u1,u2和v1,v2转换为标准形式
    构建矩阵m和向量c, 通过np.linalg.solve(m, c)解线性方程组,得到解向量
    得到交点(x, y),注意平行的话np.linalg.solve会抛出异常
    :param u1: 直线u
    :param u2:
    :param v1: 直线v
    :param v2:
    :return: 交点(x, y)
    """

    a1, b1, c1 = standard_form(u1, u2)
    a2, b2, c2 = standard_form(v1, v2)
    m = np.array(((a1, b1), (a2, b2)))
    c = np.array((c1, c2))
    return np.linalg.solve(m, c)

# Will fail if lines are parallel!
# def do_segments_intersect(s1,s2):
#     u1,u2 = s1
#     v1,v2 = s2
#     l1, l2 = distance(*s1), distance(*s2)
#     x,y = intersection(u1,u2,v1,v2)
#     return (distance(u1, (x,y)) <= l1 and
#             distance(u2, (x,y)) <= l1 and
#             distance(v1, (x,y)) <= l2 and
#             distance(v2, (x,y)) <= l2)


def segment_checks(s1, s2):
    """
    通过一组布尔值，表示交点(x,y)是否在各线段s1,s2范围内
    如果交点到每个端点的距离小于等于线段长度，则交点在该线段上
    :param s1:
    :param s2:
    :return:一组布尔值
    """
    u1, u2 = s1
    v1, v2 = s2
    l1, l2 = distance(*s1), distance(*s2)
    x, y = intersection(u1, u2, v1, v2)
    return [
        distance(u1, (x, y)) <= l1,
        distance(u2, (x, y)) <= l1,
        distance(v1, (x, y)) <= l2,
        distance(v2, (x, y)) <= l2
    ]


def do_segments_intersect(s1, s2):
    """
    判断两条线段s1, s2是否相交
    通过intersection 计算2条直线的交点(x,y)
    如果交点存在且在两条线段范围内(通过检查距离),则线段相交
    如果直线平行,np.linalg.LinAlgError捕获异常，返回Flase
    :param s1:
    :param s2:
    :return:
    """
    u1, u2 = s1
    v1, v2 = s2
    d1, d2 = distance(*s1), distance(*s2)  # 将第一条线段和第二条线段的长度分别存储为d1和d2
    try:
        x, y = intersection(u1, u2, v1, v2)  # 找出线段所在直线的交点(x, y)

        # 进行4次检查以确保交点 位于线段的4个端点之间，确认线段相交
        return (distance(u1, (x, y)) <= d1 and
                distance(u2, (x, y)) <= d1 and
                distance(v1, (x, y)) <= d2 and
                distance(v2, (x, y)) <= d2)
    except np.linalg.linalg.LinAlgError:
        return False


# print(do_segments_intersect(((0,2),(1,-1)),((0,0),(4,0))))

# a = np.array(((1,0), (0,1)))
# b = np.array((9,0))
# x = np.linalg.solve(a, b)
# print(x)
