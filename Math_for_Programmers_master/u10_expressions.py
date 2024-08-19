from abc import ABCMeta, abstractmethod


class Expressions(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, **bindings):
        """
        用于将变量绑定作为参数传入
        :param bindings: 在计算机科学术语中，这叫作变量绑定（variable binding）
        :return:
        """
        pass


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


class Sum(Expressions):
    """
    组合器: Sum 接收任意数量的输入表达式
    允许计算任意个项的和，从而可以将两个或更多表达式相加
    """
    def __init__(self, *exps):
        self.exps = exps


class Difference(Expressions):
    """
    组合器: Quotient 表示两个表达式相减
    Difference组合器需要存储两个表达式，表示从第一个表达式中减去第二个表达式
    """
    def __init__(self, exp1, exp2):
        self.exp1 = exp1
        self.exp2 = exp2


class Quotient(Expressions):
    """
    组合器: Quotient 表示两个表达式相除
    numerator: 分子
    denominator: 分母
    """
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator


class Negative(Expressions):
    """
    组合器 Negative: 表示一个表达式取反。
    """
    def __init__(self, exp):
        self.exp = exp


class Number(Expressions):
    """
    元素类(变量/数字/应用名): 数字类
    """
    def __init__(self, number):
        self.number = number

    def evaluate(self, **bindings):
        return self.number


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


class Function():
    """
    元素类(变量/数字/应用名): 应用名称,使用字符串存储函数名称（如"sin"）
    """
    def __init__(self, name):
        self.name = name


class Apply(Expressions):
    """
    元素类的顶层类Apply,同Sum一个层级
    存储一个函数(function)以及传入该函数的参数(argument)
    """
    def __init__(self, function, argument):
        self.function = function
        self.argument = argument


"""
  
(3x**2 + x) sin(x)的准确表示:
f_expression = Product(>    
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

Apply(Function("cos"),Sum(Power(Variable("x"),Number("3")), Number(-5))) 

"""


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


