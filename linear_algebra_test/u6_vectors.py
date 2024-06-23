from abc import ABCMeta, abstractmethod
from random import uniform
from math import isclose


# Base Vector class
class Vector(metaclass=ABCMeta):
    @abstractmethod
    def scale(self, scalar):
        pass

    @abstractmethod
    def add(self, other):
        pass

    @abstractmethod
    def zero(self):
        """zero 抽象方法，以返回给定向量空间中的零向量"""
        pass

    def __neg__(self):
        """用于重载取反运算符的特殊方法名 """
        return self.scale(-1)

    def __mul__(self, scalar):
        """__mul__定义了标量在向量左边的标量乘法"""
        return self.scale(scalar)

    def __rmul__(self, scalar):
        """__rmul__定义了标量在向量右边的标量乘法"""
        return self.scale(scalar)

    def __add__(self, other):
        return self.add(other)

    def __truediv__(self, scalar):
        """用向量除以标量。将向量乘以标量的倒数（1.0/标量），就可以将向量除以非零标量。"""
        return self.scale(1.0/scalar)


class Vec0(Vector):
    """
    向量空间是零维(zero-demensional)空间 R。这是坐标数为零的向量集，
    可以被描述为空的元组或继承自Vector的类Vec0。
    """
    def __init__(self):
        pass

    def add(self, other):
        assert self.__class__ == other.__class__
        return Vec0()

    def scale(self, scalar):
        return Vec0()

    @classmethod
    def zero(cls):
        return Vec0()

    def __eq__(self, other):
        return self.__class__ == other.__class__ == Vec0

    def __repr__(self):
        return "Vec0()"


class Vec1(Vector):
    """
    具有单一坐标的向量,Vec1上的向量加法和标量乘法其实等价于对其所包装的数进行加法和乘法运算。
    """
    def __init__(self, x):
        self.x = x

    def add(self, other):
        assert self.__class__ == other.__class__
        return Vec1(self.x + other.x)

    def scale(self, scalar):
        return Vec1(scalar * self.x)

    @classmethod
    def zero(cls):
        return Vec1(0)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x

    def __repr__(self):
        return "Vec1({})".format(self.x)


# Vec2 class for 2D vectors
class Vec2(Vector):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, other):
        assert self.__class__ == other.__class__
        return Vec2(self.x + other.x, self.y + other.y)

    def scale(self, scalar):
        return Vec2(scalar * self.x, scalar * self.y)

    def zero(self):
        return Vec2(0, 0)

    def __eq__(self, other):
        """运算符重载: 利用特殊方法或者魔法方法处理自定义逻辑的 == 运算符"""
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y

    def __repr__(self):
        """重写__repr__方法来改变Vec2对象的字符串表示"""
        return "Vec2({},{})".format(self.x, self.y)


# Vec3 class for 3D vectors
class Vec3(Vector):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def add(self, other):
        assert self.__class__ == other.__class__
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def scale(self, scalar):
        return Vec3(scalar * self.x, scalar * self.y, scalar * self.z)

    def zero(self):
        return Vec3(0, 0, 0)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y and self.z == other.z

    def __getitem__(self, index):
        """
        支持向量索引,用于Matrix5_by_3对象和Vec3对象,实现变量或矩阵与向量相乘
        :param index: 索引
        :return: 元组中对应索引的元素
        """
        return (self.x, self.y, self.z)[index]

    def __repr__(self):
        return "Vec3({},{},{})".format(self.x, self.y, self.z)


# CoordinateVector class for generalized coordinate vectors
class CoordinateVector(Vector):
    """
    继承自Vector 的类CoordinateVector，添加一个代表维度的抽象属性，以此节省因为坐标维度不同而带来的重复工作
    """
    @property
    @abstractmethod
    def dimension(self):
        pass

    def __init__(self, *coordinates):
        """ 初始化向量坐标从子类接收具体维度数目 """
        self.coordinates = tuple(coordinates)

    def add(self, other):
        assert self.__class__ == other.__class__
        return self.__class__(*(a + b for a, b in zip(self.coordinates, other.coordinates)))

    def scale(self, scalar):
        return self.__class__(*(scalar * x for x in self.coordinates))

    def zero(self):
        return self.__class__(*(0 for _ in self.coordinates))

    def __neg__(self):
        return self.__class__(*(-x for x in self.coordinates))

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, scalar):
        return self.scale(scalar)

    def __rmul__(self, scalar):
        return self.scale(scalar)

    def __repr__(self):
        return "{}{}".format(self.__class__.__qualname__, self.coordinates)


class Vec6(CoordinateVector):
    """定义具体维度"""
    @property
    def dimension(self):
        return 6


class LinearFunction(Vec2):
    """
    函数向量空间:f(x) = ax + b
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def add(self, v):
        return LinearFunction(self.a + v.a, self.b + v.b)

    def scale(self, scalar):
        return LinearFunction(scalar * self.a, scalar * self.b)

    def __call__(self, input):
        """
        通过__call__方法实现LinearFunction类的实力能像函数一样被调用,测试在 u6_high_matrix中
        :param input: 一个数值，用于计算线性函数的输出
        :return: 返回输入input下的结果，即 f(input) = a * input + b
        """
        return self.a * input + self.b

    @classmethod
    def zero(cls):
        return LinearFunction(0, 0, 0)


class QuadraticFunction(Vector):
    """
    实现类QuadraticFunction(Vector)
    表示ax2 + bx + c形式的函数生成的向量子空间
    """
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def add(self, v):
        return QuadraticFunction(self.a + v.a,
                                 self.b + v.b,
                                 self.c + v.c)

    def scale(self, scalar):
        return QuadraticFunction(scalar * self.a,
                                 scalar * self.b,
                                 scalar * self.c)

    def __call__(self, x):
        return self.a * x * x + self.b * x + self.c

    @classmethod
    def zero(cls):
        return QuadraticFunction(0, 0, 0)


# Test functions and random vector generators
def random_scalar():
    """使用random.uniform函数来生成−10和10之间均匀分布的浮点数"""
    return uniform(-10, 10)


def random_vec2():
    """根据随机数生成随机二维向量"""
    return Vec2(random_scalar(), random_scalar())


def approx_equal_vec2(v, w):
    """
    测试x分量和y分量是否接近(即使不相等)
    为忽略这种微小的差异,函数用来判断两个浮点数的值是否近似相等
    """
    return isclose(v.x, w.x) and isclose(v.y, w.y)


def random_vec3():
    return Vec3(random_scalar(), random_scalar(), random_scalar())


def approx_equal_vec3(v, w):
    return isclose(v.x, w.x) and isclose(v.y, w.y) and isclose(v.z, w.z)


def test(eq, a, b, u, v, w):
    """
    test函数的eq参数为具体的检验函数，这样的设计可以很好地解耦检验函数和具体的向量与标量
    测试6条规则:
    (1) 向量相加与顺序无关：v + w = w + v适用于任意v和w。
    (2) 向量相加与如何分组无关：u + (v + w)等同于(u + v) + w，这代表u + v + w是无歧义的。
    举一个经典的反例：通过+拼接字符串。在 Python 中，你可以执行"hot" + "dog"，但是字符串处理起来和向量不一样，
    因为"hot"+"dog"与"dog"+"hot"这两个和不等价，违反了规则(1)。
    标量乘法也需要是良态的，并且和加法兼容。举个例子，整数标量乘法应该等于重复加法（比如，
    3 · v = v + v + v），下面是具体的规则。
    (3) 向量和若干标量相乘等价于和这些标量之积相乘：如果 a和 b是标量，v是向量，
    那么a · (b · v)和(a · b) · v等价。
    (4) 向量和1相乘保持不变：1 · v = v。
    (5) 标量加法应该与标量乘法兼容：a · v + b · v和(a + b) · v等价。
    (6) 向量加法同样应该与标量乘法兼容：a · (v + w)和a · v + a · w 等价。
    zero()是实例方法，不应该向其他方法一样要参数，相反，他应该创建并返回相同类型的零向量，利用v.zero()实例化列零向量
    """
    zero = v.zero()
    assert eq(u + v, v + u)
    assert eq(u + (v + w), (u + v) + w)
    assert eq(a * (b * v), (a * b) * v)
    assert eq(1 * v, v)
    assert eq((a + b) * v, a * v + b * v)
    assert eq(a * v + a * w, a * (v + w))
    assert eq(zero + v, v)
    assert eq(0 * v, zero)
    assert eq(-v + v, zero)


# Example usage and tests
if __name__ == '__main__':
    all_test_passed = True

    # Test vec2
    try:
        for i in range(0, 100):
            a, b = random_scalar(), random_scalar()
            u, v, w = random_vec2(), random_vec2(), random_vec2()
            test(approx_equal_vec2, a, b, u, v, w)
        print("Vec2 test passed.")
    except AttributeError as e:
        all_test_passed = False
        print("Vec2 test failed.", e)

    # Test vec3
    try:
        for i in range(0, 100):
            a, b = random_scalar(), random_scalar()
            u, v, w = random_vec3(), random_vec3(), random_vec3()
            test(approx_equal_vec3, a, b, u, v, w)
        print("Vec3 test passed.")
    except AttributeError as e:
        all_test_passed = False
        print("Vec3 test failed.", e)

    if all_test_passed:
        print("All tests passed successfully")
    else:
        print("Some tests failed")

    # Testing CoordinateVector
    print(Vec6(1, 2, 3, 4, 5, 6) + Vec6(1, 2, 3, 4, 5, 6))
    print(Vec6(1, 2, 3, 4, 5, 6) * 4)
    print(Vec6(1, 2, 3, 4, 5, 6) * 0)
    print(Vec6(1, 2, 3, 4, 5, 6) / 2)





