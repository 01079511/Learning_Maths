from datetime import datetime
from json import loads
from pathlib import Path
from u6_vectors import Vector, Vec3, LinearFunction
import numpy as np
import matplotlib.pyplot as plt


# 识别现实中的向量
class CarForSale(Vector):
    """
    二手丰田普锐斯的数据集作为向量处理的类
    """
    # 1于2018 年 11 月 30日正午从CarGraph网站获取到的数据集
    retrieved_date = datetime(2018, 11, 30, 12)

    def __init__(self, model_year, mileage, price, posted_datetime,
                 model="(virtual)", source="(virtual)",
                 location="(virtual)", description="(virtual)"):
        """
        2 为了简化构造函数
        所有的字符串参数都是可选的，默认值为"(virtual)
        """
        self.model_year = model_year
        self.mileage = mileage
        self.price = price
        self.posted_datetime = posted_datetime
        self.model = model
        self.source = source
        self.location = location
        self.description = description

    def add(self, other):
        def add_dates(d1, d2):
            """
            3 工具函数
            通过叠加时间跨度来实现将日期相加
            """
            age1 = CarForSale.retrieved_date - d1
            age2 = CarForSale.retrieved_date - d2
            sum_age = age1 + age2
            return CarForSale.retrieved_date - sum_age

        #  4通过对属性求和来生成新的CarForSale实例
        return CarForSale(
            self.model_year + other.model_year,
            self.mileage + other.mileage,
            self.price + other.price,
            add_dates(self.posted_datetime, other.posted_datetime)
        )

    def scale(self, scalar):
        def scale_date(d):
            """
            5 工具函数
            根据传入的数值来缩放时间跨度
            """
            age = CarForSale.retrieved_date - d
            return CarForSale.retrieved_date - (scalar * age)
        return CarForSale(
            scalar * self.model_year,
            scalar * self.mileage,
            scalar * self.price,
            scale_date(self.posted_datetime)
        )

    @classmethod
    def zero(cls):
        """
        :return: 0向量的实例
        """
        return CarForSale(0, 0, 0, CarForSale.retrieved_date)


# 将矩阵作为向量处理,Matrix是抽象基类
class Matrix(Vector):
    @property
    def rows(self):
        """
        抽象属性
        :return:行数
        """
        pass

    @property
    def columns(self):
        """
        抽象属性
        :return:列数
        """
        pass

    def __init__(self, entries):
        """
        构造函数
        :param entries: 嵌套的元组，表示矩阵的元素，将传入的entries,存储在实例变量self.entries中
        """
        self.entries = entries

    def add(self, other):
        """
        向量相乘
        :param other:
        :return:
        for i in range(0, self.rows()) 遍历矩阵的每一行
        for j in range(0, self.columns())遍历矩阵的每一列
        """
        return self.__class__(
            tuple(
                tuple(self.entries[i][j] + other.entries[i][j]
                        for j in range(0, self.columns()))
                for i in range(0, self.rows())))

    def scale(self, scalar):
        """
        标量相乘
        :param scalar:
        :return:
        for row in self.entries 遍历矩阵的每一行
        for e in row 遍历矩阵每一行中的每一个元素
        scalar * x 每个元素与标量相乘
        """
        return self.__class__(
            tuple(
                tuple(scalar * e for e in row)
                for row in self.entries))

    def __repr__(self):
        """
        表示方法
        :return: 包含类名和矩阵元素的字符串表示，便于调用和查看内容
        """
        return "%s%r" % (self.__class__.__qualname__, self.entries)

    def zero(self):
        """
        零矩阵方法
        :return:返回一个新的Matrix对象，包含一个全零的嵌套元组
        """
        return self.__class__(
            tuple(
                tuple(0 for i in range(0, self.columns()))
                for j in range(0, self.rows())))


class Matrix2By2(Matrix):
    def rows(self):
        return 2

    def columns(self):
        return 2


class Matrix5By3(Vector):
    # 定义类属性--行列数,构筑 0矩阵
    rows = 5
    columns = 3

    def __init__(self, entries):
        self.entries = entries

    def add(self, other):
        """
        矩阵相加
        :param other: 2个Matrix5By3对象
        :return: 相加后的结果
        """
        return Matrix5By3(tuple(
            tuple(a + b for a, b in zip(row1, row2))
            for (row1, row2) in zip(self.entries, other.entries)
        ))

    def scale(self, scalar):
        """
        标量相乘
        :param scalar: Matrix5By3对象和标量scalar相乘
        :return: 标量相乘后Matrix5By3对象
        """
        return Matrix5By3(tuple(
            tuple(scalar * x for x in row)
            for row in self.entries
        ))

    def __mul__(self, other):
        """
        对向量和矩阵类的*运算符进行重载,用于Matrix5_by_3对象和Vec3对象,实现变量或矩阵与向量相乘
        :param other:
        :return:
        """
        if not isinstance(other, Vec3):
            raise TypeError("Matrix5By3 can only be multiplied by Vec3")
        result = [sum(self.entries[i][j] * other[j]
                      for j in range(self.columns))
                  for i in range(self.rows)]
        return result

    @classmethod
    def zero(cls):
        return Matrix5By3(tuple(  # 5 × 3矩阵的零向量是一个全部由0组成的5 × 3矩阵。把它和任意5 × 3矩阵M相加，都会返回M
            tuple(0 for j in range(0, cls.columns))
            for i in range(0, cls.rows)
        ))


# 函数作为向量处理
class Function(Vector):

    def __init__(self, f):
        self.function = f

    # 向量加法
    def add(self, other):
        return Function(lambda x: self.function(x) + other.function(x))

    # 标量相乘
    def scale(self, scalar):
        return Function(lambda x: scalar * self.function(x))

    @classmethod
    def zero(cls):
        return Function(lambda x: 0)

    # 类的实例可以像调用函数一样被调用
    def __call__(self, arg):
        return self.function(arg)


class Function2(Vector):
    def __init__(self, f):
        self.function = f

    def add(self, other):
        return Function2(lambda x, y: self.function(x, y) + other.function(x, y))

    def scale(self, scalar):
        return Function2(lambda x, y: scalar * self.function(x, y))

    @classmethod
    def zero(cls):
        return Function2(lambda x, y: 0)

    # 类的实例可以像调用函数一样被调用
    def __call__(self, *args):
        return self.function(*args)


if __name__ == '__main__':

    # ETL过程开始(extract)
    # load cargraph data from json file
    contents = Path('cargraph.json').read_text()
    # 使用json.loads将JSON字符串解析为Python对象，存储在变量cg中
    cg = loads(contents)
    # 初始化一个空列表命名为cleaned,存储清洗后的CarForSale对象
    cleaned = []

    def parse_date(s):
        """
        解析日期函数
        :param s:日期字符串
        :return:通过input_format解析日期返回一个datetime对象,格式: %m/%d - %H:%M
        """
        input_format = "%m/%d - %H:%M"
        return datetime.strptime(s, input_format).replace(year=2018)

    # Transform: 清洗和转换数据,遍历cg中每一行数据(从第二行开始,跳过表头)
    for car in cg[1:]:
        try:
            # 尝试将每行数据转换为CarForSale对象,如int(car[1])是将mileage转换为整数
            row = CarForSale(int(car[1]), float(car[3]), float(car[4]), parse_date(car[6]), car[2], car[5], car[7],
                             car[8])
            # 如果转换成功将CarForSale对象row添加到cleaned列表中
            cleaned.append(row)
        except:
            pass

    # ETL过程结束(load)
    # 包含清晰和转换后的CarForSale对象
    cars = cleaned

    # sum调用CarForSale类的__add__方法把每个对象的属性值加在一起，结果是所有cars对象相应属性值的和,CarForSale.zero()作用作为sum()的初始值以及确定返回值的类型为CarForSale对象
    # / 调用CarForSale类的__mul__方法累加结果除以cars列表长度即二手车数量 = 每个属性的平均值
    average_prius = sum(cars, CarForSale.zero()) * (1.0 / len(cars))
    # 以字典形式打印结果
    print("Cars数据集分析结果")
    print(average_prius.__dict__)

    # 定义绘图函数,可视化函数作为向量的结果
    def plot(fs, xmin, xmax):
        xs = np.linspace(xmin, xmax, 100)  # 生成从xmin->xmax的100个点
        fig, ax = plt.subplots()  # 创建一个图形和子图
        ax.axhline(y=0, color='k')  # 在y=0位置画一条水平线，颜色为黑色
        ax.axvline(x=0, color='k')  # 在x=0位置画一条水平线，颜色为黑色
        for f in fs:
            ys = [f(x) for x in xs]  # 计算每个x点对应的y值
            plt.plot(xs, ys)  # 绘图
        # plt.legend()  # 显示图例
        plt.show()  # 显示图形
        plt.close()  # 关闭图形窗口，释放资源

    # 定义函数f & g and f2 & g2
    f = Function(lambda x: 0.5 * x + 3)
    g = Function(np.sin)
    f2 = Function2(lambda x, y: x + y)
    g2 = Function2(lambda x, y: x - y + 1)

    # 调用绘图函数
    # plot([f, g, f + g, 3 * g], -10, 10)
    # 调用绘图函数，测试线性函数
    plot([LinearFunction(-2, 2)], -5, 5)

    # 测试Function2
    print("测试Function2")
    print((f2 + g2)(3, 10))  # 7

    # 测试Matrix
    print("测试Matrix")
    print(2 * Matrix2By2(((1, 2), (3, 4))) + Matrix2By2(((1, 2), (3, 4))))  # Matrix2by2((3, 6), (9, 12))

    # 测试Matrix5by3对象和Vec3对象运行矩阵乘法
    print("测试Matrix5by3对象和Vec3对象运行矩阵乘法")
    m1 = Matrix5By3([(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15)])
    v1 = Vec3(1, 2, 3)
    print(m1 * v1)
