from math import isclose
from random import uniform
from u6_vectors import random_scalar, test
from u6_high_matrix import Matrix5By3


def random_matrix(rows, columns):
    return tuple(
        tuple(uniform(-10, 10) for j in range(0, columns))
    for i in range(0, rows)
    )


def random_5_by_3():
    return Matrix5By3(random_matrix(5, 3))


def approx_equal_matrix_5_by_3(m1, m2):
    return all([
        isclose(m1.entries[i][j], m2.entries[i][j])
        for j in range(0, 3)
        for i in range(0, 5)
    ])


for i in range(0, 100):
    a, b = random_scalar(), random_scalar()
    u, v, w = random_5_by_3(), random_5_by_3(), random_5_by_3()
    test(approx_equal_matrix_5_by_3, a, b, u, v, w)


