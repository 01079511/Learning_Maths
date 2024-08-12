import pygame
import u7_vectors as vectors
from math import pi, sqrt, cos, sin, atan2
from random import randint, uniform
from u7_linear_solver import do_segments_intersect
import sys


class PolygonModel():
    """
    PolygonModel类: 代表一个可以平移或旋转并保持形状不变的游戏实体(飞船或小行星)
    """
    def __init__(self, points):
        """
        当飞船或小行星移动时，通过self.x和self.y进行平移，
        并通过self.rotation_angle进行旋转，可以找出其实际位置
        vx和vy属性存储了vx = x'(t)和vy = y'(t)当前的值。默认情况下，它们的值为0，表示对象没有运动
        :param points:
        """
        self.points = points
        self.rotation_angle = 0
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.angular_velocity = 0
        self.draw_center = False

    def transformed(self):
        """
        用于计算多边形在经过旋转和平移后的变换点(该方法返回由对象的 x 属性和 y 属性转换并由 rotation_angle 属性旋转的模型的点)
        它将多边形的每个点首先按照指定角度旋转，然后平移到指定位置
        :return: 包含旋转和平移后的点
        """
        # 旋转遍历生成都多边形的每个点，通过rotate2d将其绕原点旋转rotation_angle角度
        rotated = [vectors.rotate2d(self.rotation_angle, v) for v in self.points]  # 包含旋转后的列表
        # 平移,通过add(self.x, self.y), v)平移self.x, self.y
        return [vectors.add((self.x, self.y), v) for v in rotated]

    def segments(self):
        """
        以避免重复返回构成多边形的（已变换的）线段，这很方便。
        然后，我们可以检查另一个多边形的每一条线段，
        看它与当前多边形的does_intersect是否返回True
        :return:
        """
        point_count = len(self.points)
        points = self.transformed()
        return [(points[i], points[(i+1) % point_count])
                for i in range(0, point_count)]

    def does_collide(self, other_poly):
        """
        实现does_collide(other_polygon)方法，
        通过检查定义两个多边形的任何线段是否相交,
        来确定当前PolygonModel对象是否与other_polygon发生碰撞。
        这可以帮助我们确定小行星是撞击了飞船还是撞击了另一颗小行星。
        :param other_poly:
        :return:
        """
        for other_segment in other_poly.segments():
            if self.does_intersect(other_segment):
                return True
        return False

    def does_intersect(self, other_segment):
        """
        检查多边形是否与另一线段(如原点射出的射线)相交
        :param other_segment:另一个线段
        :return:布尔值，表示是否相交
        """
        for segment in self.segments():
            # 如果多边形的任何一条线段与other_segment相交，则该方法返回True
            if do_segments_intersect(other_segment, segment):
                return True
        return False

    def move(self, milliseconds):
        dx, dy = (self.vx * milliseconds / 1000.0,
                  self.vy * milliseconds / 1000.0)
        self.x, self.y = vectors.add((self.x, self.y),
                                     (dx, dy))
        if self.x < -10:
            self.x += 20
        if self.y < -10:
            self.y += 20
        if self.x > 10:
            self.x -= 20
        if self.y > 10:
            self.y -= 20
        self.rotation_angle += self.angular_velocity * milliseconds / 1000.0


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

    def laser_segment(self):
        """
        在二维世界中，激光束应该是一条线段，从经过变换的宇宙飞船顶端开始，向飞船指向的方向延伸
        :return:
        """
        dist = 20. * sqrt(2)  # 使用勾股定理找到屏幕上的最长线段
        x, y = self.transformed()[0]  # 获取定义线段的第一点(飞船的顶端)的值
        return ((x, y),
                (x + dist * cos(self.rotation_angle),
                 y + dist*sin(self.rotation_angle)))  # 如果激光以角度 self.rotation_angle 从顶端(x, y)延伸dist 单位，则使用三角函数找到激光的终点


class Asteroid(PolygonModel):
    """
    小行星实例是 PolygonModel 的具体例子
    """
    def __init__(self):
        """
        vx和vy,表示将x和y 方向上的速度设置为–1和1之间的随机值
        为负值的导数意味着函数值在减小，而正值意味着函数值在增大
        """
        sides = randint(5, 9)  # 行星的边数是5和9之间的一个随机整数
        vs = [vectors.to_cartesian((uniform(0.5, 1.0), 2 * pi * i / sides))
                for i in range(0, sides)]  # 长度是0.5和1.0之间的随机浮点数,角度是2π/n的倍数,其中n是边数
        super().__init__(vs)
        self.vx = uniform(-1, 1)
        self.vy = uniform(-1, 1)


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
    :return:对应的屏幕像素坐标(px,py)
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

# asteroid = PolygonModel([(2, 7), (1, 5), (2, 3), (4, 2), (6, 2), (7, 4), (6, 6), (4, 6)])
# print(asteroid.does_intersect([(0, 0), (7, 7)]))


def draw_segment(screen, v1, v2, color=RED):
    """
    2024.08.06
    :param screen:
    :param v1:
    :param v2:
    :param color:
    :return:
    """
    pygame.draw.aaline(screen, color, to_pixels(*v1), to_pixels(*v2), 10)


screenshot_mode = False

# INITIALIZE GAME ENGINE


def main():
    """
    2024.08.06
    """
    pygame.init()

    screen = pygame.display.set_mode([width,height])

    pygame.display.set_caption("Asteroids!")

    done = False
    clock = pygame.time.Clock()

    # p key prints screenshot (you can ignore this variable)
    p_pressed = False

    while not done:

        clock.tick()

        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        # UPDATE THE GAME STATE

        milliseconds = clock.get_time()
        keys = pygame.key.get_pressed()

        for ast in asteroids:
            ast.move(milliseconds)

        if keys[pygame.K_LEFT]:
            ship.rotation_angle += milliseconds * (2*pi / 1000)

        if keys[pygame.K_RIGHT]:
            ship.rotation_angle -= milliseconds * (2*pi / 1000)

        laser = ship.laser_segment()

        # p key saves screenshot (you can ignore this)
        if keys[pygame.K_p] and screenshot_mode:
            p_pressed = True
        elif p_pressed:
            pygame.image.save(screen, 'figures/asteroid_screenshot_%d.png' % milliseconds)
            p_pressed = False

        # DRAW THE SCENE

        screen.fill(WHITE)

        if keys[pygame.K_SPACE]:
            draw_segment(screen, *laser)

        draw_poly(screen,ship)

        for asteroid in asteroids:
            if keys[pygame.K_SPACE] and asteroid.does_intersect(laser):
                asteroids.remove(asteroid)
            else:
                draw_poly(screen, asteroid, color=GREEN)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    if '--screenshot' in sys.argv:
        screenshot_mode = True
    main()







