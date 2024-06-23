from math import isclose
from random import uniform, random, randint
from datetime import datetime, timedelta
from u6_high_matrix import CarForSale
from u6_vectors import random_scalar, test


# 对CarForSale 进行向量空间的单元测试，证明它的对象形成了一个向量空间（忽略其文本属性）
def random_time():
    # timedelta(days=uniform(0, 10))是0-10的随机天数
    return CarForSale.retrieved_date - timedelta(days=uniform(0, 10))


def approx_equal_time(t1, t2):
    test = datetime.now()
    # 计算t1,t2与当前时间的差值，并使用isclose函数比较它们总描述是否接近
    return isclose((test - t1).total_seconds(), (test - t2).total_seconds())


def random_car():
    """
    随机汽车生成函数
    :return: CarForSale对象
    model_year:randint(1990, 2019) 随即年份
    mileage:randint(0, 250000) 随机里程
    price: 固定价格27000
    posted_datetime:random_time() 获取随机时间
    """
    return CarForSale(randint(1990, 2019), randint(0, 250000),
                      27000. * random(), random_time())


def approx_equal_car(c1, c2):
    return (isclose(c1.model_year, c2.model_year)
            and isclose(c1.mileage, c2.mileage)
            and isclose(c1.price, c2.price)
            and approx_equal_time(c1.posted_datetime, c2.posted_datetime))


for i in range(0, 100):
    a, b = random_scalar(), random_scalar()
    u, v, w = random_car(), random_car(), random_car()
    test(approx_equal_car, a, b, u, v, w)
