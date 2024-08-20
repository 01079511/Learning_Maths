from abc import ABCMeta, abstractmethod
import math


class Expressions(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, **bindings):
        """
        抽象方法: 用于将变量绑定作为参数传入
        :param bindings: 在计算机科学术语中，这叫作变量绑定（variable binding）
        :return:
        """
        pass

    @abstractmethod
    def expand(self):
        """
        抽象方法: 实现如实例展开求和运算时，需要展开其中的每一项并把它们相加
        :return:
        """
        pass

    @abstractmethod
    def display(self):
        """
        实现将函数展开: 正确地将(a + b) (x + y)展开为 ax + ay + bx + by
        :return:
        """
        pass

    def __repr__(self):
        return self.display()


class Power(Expressions):
    """
    组合器: 表示幂的Python类
    base: 基, 如: x
    exponent: 幂, 如: 2
    即 x^2
    """
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent

    def evaluate(self, **bindings):
        return self.base.evaluate(**bindings) ** self.exponent.evaluate(**bindings)

    def expand(self):
        return self

    def display(self):
        return "Power({},{})".format(self.base.display(), self.exponent.display())


class Product(Expressions):
    """
    组合器: 存储两个相乘表达式的类
    """
    def __init__(self, exp1, exp2):
        self.exp1 = exp1
        self.exp2 = exp2

    def evaluate(self, **bindings):
        """
        在乘积过程中不需要进行替换，但我们会将绑定关系传递给两个子表达式，以防其中包含Variable
        :param bindings:
        :return:
        """
        return self.exp1.evaluate(**bindings) * self.exp2.evaluate(**bindings)

    def expand(self):
        """
        为了实现分配律，我们要处理三种情况：
        乘积的第一项可能是求和，
        第二项可能是求和，或者它们都不是求和。
        在最后一种情况下，不需要展开
        :return:
        """
        expanded1 = self.exp1.expand  # 展开乘积的两个项
        expanded2 = self.exp2.expand
        if isinstance(expanded1, Sum):
            # 如果乘积的第一个项是求和，则取其中的每项与乘积的第二个项相乘，然后在得到的结果上也调用expand方法，以防第二项也是求和
            return Sum(*[Product(e, expanded2).expand()
                         for e in expanded1.exps])
        if isinstance(expanded2, Sum):
            # 如果乘积的第二项是求和，那么就把它的每项与第一项相乘
            return Sum(*[Product(expanded1, e).expand()
                         for e in expanded2.exps])
        else:
            # 如果两项都不是求和，就不需要使用分配律
            return Product(expanded1, expanded2)

    def display(self):
        return "Product({}, {})".format(self.exp1.display(), self.exp2.display())


class Sum(Expressions):
    """
    组合器: Sum 接收任意数量的输入表达式
    允许计算任意个项的和，从而可以将两个或更多表达式相加
    """
    def __init__(self, *exps):
        self.exps = exps

    def evaluate(self, **bindings):
        return sum([exp.evaluate(**bindings) for exp in self.exps])

    def expand(self):
        return sum(*[exp.expand() for exp in self.exps])

    def display(self):
        return "Sum({})".format(",".join([e.display() for e in self.exps]))


class Difference(Expressions):
    """
    组合器: Quotient 表示两个表达式相减
    Difference组合器需要存储两个表达式，表示从第一个表达式中减去第二个表达式
    """
    def __init__(self, exp1, exp2):
        self.exp1 = exp1
        self.exp2 = exp2

    def evaluate(self, **bindings):
        return self.exp1.evaluate(**bindings) - self.exp2.evaluate(**bindings)

    def display(self):
        return "Difference({},{})".format(self.exp1.display(), self.exp2.display())


class Quotient(Expressions):
    """
    组合器: Quotient 表示两个表达式相除
    numerator: 分子
    denominator: 分母
    """
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def evaluate(self, **bindings):
        return self.numerator.evaluate(**bindings) / self.denominator.evaluate(**bindings)

    def display(self):
        return "Quotient({},{})".format(self.numerator.display(), self.denominator.display())


class Negative(Expressions):
    """
    组合器 Negative: 表示一个表达式取反。
    """
    def __init__(self, exp):
        self.exp = exp

    def evaluate(self, **bindings):
        return - self.exp.evaluate(**bindings)

    def expand(self):
        return self

    def display(self):
        return "Negative({})".format(self.exp.display())


class Number(Expressions):
    """
    元素类(变量/数字/应用名): 数字类
    """
    def __init__(self, number):
        self.number = number

    def evaluate(self, **bindings):
        return self.number

    def expand(self):
        return self

    def display(self):
        return "Number({})".format(self.number)


class Variable(Expressions):
    """
    元素类(变量/数字/应用名): 变量类
    """
    def __init__(self, symbol):
        self.symbol = symbol

    def evaluate(self, **bindings):
        try:
            return bindings[self.symbol]  # bindings是字典格式，如 x= 10 ==> bindings['x'] = 10
        except:
            raise KeyError("Variable '{}' is not bound.".format(self.symbol))

    def expand(self):
        return self

    def display(self):
        return "Variable(\"{}\")".format(self.symbol)


class Function():
    """
    元素类(变量/数字/应用名): 应用名称,使用字符串存储函数名称（如"sin"）
    """
    def __init__(self, name, make_latex=None):
        self.name = name
        self.make_latex = make_latex

    def latex(self, arg_latex):
        """
        待明确 2024.08.20
        :param arg_latex:
        :return:
        """
        if self.make_latex:
            return self.make_latex(arg_latex)
        else:
            return " \\operatorname{{ {} }} \\left( {} \\right)".format(self.name, arg_latex)


class Apply(Expressions):
    """
    元素类的顶层类Apply,同Sum一个层级
    存储一个函数(function)以及传入该函数的参数(argument)
    """
    def __init__(self, function, argument):
        self.function = function
        self.argument = argument

    def evaluate(self, **bindings):
        return _function_bindings[self.function.name](self.argument.evaluate(**bindings))

    def expand(self):
        """
        我们不能展开Apply 函数本身，但是可以展开它的参数。这将把sin(x(y + z))展开为 sin(xy + xz)
        :return:
        """
        return Apply(self.function, self.argument.expand())

    def display(self):
        return "Apply(Function(\"{}\"),{})".format(self.function.name, self.argument.display())


# 在Apply类上维护一个已知函数的字典数据, 单独_表示该部分是私有属性,不被from import引用
_function_bindings = {
    "sin": math.sin,
    "cos": math.cos,
    "ln": math.log,
    "sqrt": math.sqrt
}

_function_python = {
    "sin": "math.sin({})",
    "cos": "math.cos({})",
    "ln": "math.log({})",
    "sqrt": "math.sqrt({})"
}


# (3x**2 + x) sin(x)的准确表示(P311-图10-11):
f_expression = Product(
                Sum(
                    Product(
                        Number(3),
                        Power(
                            Variable("x"),
                            Number(2))),
                    Variable("x")),
                Apply(
                    Function("sin"),
                    Variable("x")))

# Apply(Function("cos"),Sum(Power(Variable("x"), Number("3")), Number(-5)))


def distinct_variables(exp):
    """
    接收一个表达式并返回其中不同变量的列表。
    例如，h(z) = 2z + 3包含输入变量z，
    而g(x) = 7不包含任何变量
    :param exp: 表达式(指任何元素或组合器)
    :return: 1个包含变量的Python集合
    """
    if isinstance(exp, Variable):
        return set(exp.symbol)
    elif isinstance(exp, Number):
        return set()
    elif isinstance(exp, Sum):
        return set().union(*[distinct_variables(exp) for exp in exp.exps])
    elif isinstance(exp, Product):
        return distinct_variables(exp.exp1).union(distinct_variables(exp.exp2))
    elif isinstance(exp, Power):
        return distinct_variables(exp.base).union(distinct_variables(exp.exponent))
    elif isinstance(exp, Apply):
        return distinct_variables(exp.argument)
    else:
        raise TypeError("Not a valid expression.")


# 测试
def f(x):
    return (3*x**2 + x) * math.sin(x)


print("f(5) ={0},\nf_expression.evaluate(x=5) = {1}".format(f(5), f_expression.evaluate(x=5) ))

# 测试 display()
Y = Variable('y')
Z = Variable('z')
A = Variable('a')
B = Variable('b')
print(Product(Sum(A, B), Sum(Y, Z)))
print(Product(Sum(A, B), Sum(Y, Z)).expand())
# print(f_expression.expand())

"""
Traceback (most recent call last):
  File "D:\PycharmProjects\Maths_test\Math_for_Programmers_master\u10_expressions.py", line 315, in <module>
    print(Product(Sum(A, B), Sum(Y, Z)).expand())
  File "D:\PycharmProjects\Maths_test\Math_for_Programmers_master\u10_expressions.py", line 32, in __repr__
    return self.display()
  File "D:\PycharmProjects\Maths_test\Math_for_Programmers_master\u10_expressions.py", line 95, in display
    return "Product({}, {})".format(self.exp1.display(), self.exp2.display())
AttributeError: 'function' object has no attribute 'display'
"""



