import pygame
import v7_vectors as vectors
from math import pi, sqrt, cos, sin, atan2
from random import randint, uniform
import sys


class PolygonModel():
    """
    PolygonModel类: 代表一个可以平移或旋转并保持形状不变的游戏实体(飞船或小行星)
    """
    def __init__(self, points):
        """
        当飞船或小行星移动时，通过self.x和self.y进行平移，
        并通过self.rotation_angle进行旋转，可以找出其实际位置
        :param points:
        """
        self.points = points
        self.rotation_angle = 0
        self.x = 0
        self.y = 0


class Ship(PolygonModel):
    """
    宇宙飞船实例是 PolygonModel 的具体例子
    """
    def __init__(self):
        """
        飞船具有固定的三角形形状，由3个点给出
        """
        super().__init__([(0.5, 0),
                          (-0.25, 0.25),
                          (-0.25, -0.25)])


class Asteroid(PolygonModel):
    """
    小行星实例是 PolygonModel 的具体例子
    """
    def __init__(self):
        sides = randint(5, 9)  # 行星的边数是5和9之间的一个随机整数
        vs = [vectors.to_cartesian((uniform(0.5, 1.0), 2 * pi * i / sides))
                for i in range(0, sides)]  # 长度是0.5和1.0之间的随机浮点数,角度是2π/n的倍数,其中n是边数
        super().__init__(vs)


# INITIALIZE GAME STATE:游戏的初始状态，需要一艘飞船和几颗小行星
ship = Ship()

# 开始时飞船在屏幕中心，小行星则随机分布在屏幕上。可以显示一个在x方向和y方向上分别为−10到10的平面区域
asteroid_count = 10  # 创建指定数量的Asteroid 对象的列表，在本例中数量为10
asteroids = [Asteroid() for _ in range(0, asteroid_count)]

# 将每个对象的位置设置为坐标在−10和10之间的随机点，将其显示在屏幕上
for ast in asteroids:
    ast.x = randint(-9, 9)
    ast.y = randint(-9, 9)


# HELPERS / SETTINGS
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

# 屏幕像素
width, height = 400, 400


def to_pixels(x, y):
    """
    将坐标从我们的坐标系映射成PyGame的像素坐标
    :param x:
    :param y:
    :return:
    """
    return width/2 + width * x / 20, height/2 - height * y / 20


def draw_poly(screen, polygon_model, color=GREEN):
    """
    绘制连接给定点和指定 PyGame 对象的线，参数(closed)True 指定了连接第一个点和最后一个点来创建一个闭合多边形
    :param screen:
    :param polygon_model:
    :param color:
    :return:
    """
    pixel_points = [to_pixels(x, y) for x, y in polygon_model.transformed()]
    pygame.draw.aalines(screen, color, True, pixel_points, 10)



