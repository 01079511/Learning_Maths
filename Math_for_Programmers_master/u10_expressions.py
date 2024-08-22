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

    @abstractmethod
    def _python_expr(self):
        """
        使用Python中的eval函数将其转化为可执行的Python函数。
        将结果与evaluate方法进行比较。例如，Power(Variable("x"),Number(2))表示表达式x2。
        这应该产生Python代码 x**2。然后使用Python的eval函数来执行这段代码，
        并查看它与evaluate方法的结果是否匹配
        :return:
        """
        pass

    def python_function(self, **bindings):
        #         code = "lambda {}:{}".format(
        #             ", ".join(sorted(distinct_variables(self))),
        #             self._python_expr())
        #         print(code)
        global_vars = {"math": math}
        return eval(self._python_expr(), global_vars, bindings)

    @abstractmethod
    def derevative(self, var):
        """
        对变量 var 求导数
        :param var: 变量
        :return: 导数值
        """
        pass

    @abstractmethod
    def substitute(self, var, expression):
        """
        2024.08.22 辅助Power.derevative()
        :param var:
        :param expression:
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

    def evaluate(self, **bindings):
        return self.base.evaluate(**bindings) ** self.exponent.evaluate(**bindings)

    def expand(self):
        return self

    def display(self):
        return "Power({},{})".format(self.base.display(), self.exponent.display())

    def _python_expr(self):
        return "({}) ** ({})".format(self.base._python_expr(), self.exponent._python_expr())

    def substitute(self, var, exp):
        return Power(self.base.substitute(var, exp), self.exponent.substitute(var, exp))

    def derivative(self, var):
        if isinstance(self.exponent, Number):  # 如果指数是一个数，使用幂法则
            power_rule = Product(
                    Number(self.exponent.number),
                    Power(self.base, Number(self.exponent.number - 1)))
            return Product(self.base.derivative(var), power_rule)  # f(x)**n的导数是f'(x) · nf(x)**n – 1，所以这里根据链式法则乘以f'(x)
        elif isinstance(self.base, Number):  # 检查基数是否为数：如果是，我们使用指数法则
            exponential_rule = Product(Apply(Function("ln"), Number(self.base.number)), self)
            return Product(self.exponent.derivative(var), exponential_rule)  # 如果要求 a**f(x)的导数，那么同样根据链式法则，乘以f'(x)的系数
        else:  # 当基数和指数都不是数时，会抛出一个异常
            raise Exception("couldn't take derivative of power {}".format(self.display()))


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
        expanded1 = self.exp1.expand()  # 展开乘积的两个项
        expanded2 = self.exp2.expand()
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

    def _python_expr(self):
        return "({})*({})".format(self.exp1._python_expr(), self.exp2._python_expr())

    def substitute(self, var, exp):
        return Product(self.exp1.substitute(var, exp), self.exp2.substitute(var, exp))

    def derevative(self, var):
        return  Sum(
            Product(self.exp1.derevative(var), self.exp2),
            Product(self.exp1, self.exp2.derevative(var))
        )


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
        return Sum(*[exp.expand() for exp in self.exps])

    def display(self):
        return "Sum({})".format(",".join([e.display() for e in self.exps]))

    def _python_expr(self):
        return "+".join("({})".format(exp._python_expr()) for exp in self.exps)

    def derevative(self, var):
        return Sum(*[exp.derevative(var) for exp in self.exps])

    def substitute(self, var, new):
        return Sum(*[exp.substitute(var,new) for exp in self.exps])


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

    def _python_expr(self):
        return "({}) - ({})".format(self.exp1._python_expr(), self.exp2._python_expr())

    def substitute(self, var, exp):
        return Difference(self.exp1.substitute(var, exp), self.exp2.substitute(var, exp))


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

    def _python_expr(self):
        return "({}) / ({})".format(self.exp1._python_expr(), self.exp2._python_expr())

    def substitute(self, var, exp):
        return Quotient(self.numerator.substitute(var, exp), self.denominator.substitute(var, exp))


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

    def _python_expr(self):
        return "- ({})".format(self.exp._python_expr())

    def substitute(self,var,exp):
        return Negative(self.exp.substitute(var,exp))


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

    def _python_expr(self):
        return str(self.number)

    def derevative(self, var):
        return Number(0)

    def substitute(self, var, exp):
        return self


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

    def _python_expr(self):
        return self.symbol

    def substitute(self, var, exp):
        if self.symbol == var.symbol:
            return exp
        else:
            return self

    def derevative(self, var):
        """
        只有当一个变量是我们要进行导数计算的变量时，它的导数才是1，否则导数就是0,
        如果求f(x) = x的导数，结果是f'(x) = 1，对应的就是该直线的斜率。
        求f(x) = c的导数应该得到0，因为c在这里代表一个常数，而不是函数f的参数
        :param var: 变量
        :return: 导数值
        """
        if self.symbol == var.symbol:
            return Number(1)
        else:
            return Number(0)


class Function():
    """
    元素类(变量/数字/应用名): 应用名称,使用字符串存储函数名称（如"sin"）
    """
    def __init__(self, name, make_latex=None):
        self.name = name
        self.make_latex = make_latex


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

    def _python_expr(self):
        return _function_python[self.function.name].format(self.argument._python_expr())

    def substitute(self, var, exp):
        return Apply(self.function, self.argument.substitute(var, exp))

    def derivative(self, var):
        return Product(
                self.argument.derivative(var),
                _derivatives[self.function.name].substitute(_var, self.argument))

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

# 创建一个占位符，这样就不会与实际使用的其他符号(如x或y)混淆
_var = Variable('placeholder variable')

# 编码一些特殊函数的导数，它不能与我们在实践中使用的变量冲突。导数被存储为一个从函数名到其导数表达式的字典映射
_derivatives = {
    "sin": Apply(Function("cos"), _var),
    "cos": Product(Number(-1), Apply(Function("sin"), _var)),
    "ln": Quotient(Number(1), _var),
    "sqrt": Quotient(Number(1), Product(Number(2), Apply(Function("sqrt"), _var)))
}


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


def contains(exp, var):
    """
    函数 contains(expression, variable)，检查给定的表达式是否包含指定的变量
    通过 distinct_variables 的结果中, 可以很容易地检查变量是否存在
    :param exp: 表达式
    :param var: 指定的变量
    :return:
    """
    if isinstance(exp, Variable):
        return exp.symbol == var.symbol
    elif isinstance(exp, Number):
        return False
    elif isinstance(exp, Sum):
        return any([contains(e, var) for e in exp.exps])
    elif isinstance(exp, Product):
        return contains(exp.exp1, var) or contains(exp.exp2, var)
    elif isinstance(exp, Power):
        return contains(exp.base, var) or contains(exp.exponent, var)
    elif isinstance(exp, Apply):
        return contains(exp.argument, var)
    else:
        raise TypeError("Not a valid expression.")


def distinct_functions(exp):
    """
    函数distinct_functions，接收一个表达式作为参数，并返回表达式中不重复的函数名(如sin或ln)
    :param exp:表达式作为参数
    :return:表达式中不重复的函数名(如sin或ln)
    """
    if isinstance(exp, Variable):
        return set()
    elif isinstance(exp, Number):
        return set()
    elif isinstance(exp, Sum):
        return set().union(*[distinct_functions(exp) for exp in exp.exps])
    elif isinstance(exp, Product):
        return distinct_functions(exp.exp1).union(distinct_functions(exp.exp2))
    elif isinstance(exp, Power):
        return distinct_functions(exp.base).union(distinct_functions(exp.exponent))
    elif isinstance(exp, Apply):
        return set([exp.function.name]).union(distinct_functions(exp.argument))
    else:
        raise TypeError("Not a valid expression.")


def contains_sum(exp):
    """
    函数contains_sum，接收一个表达式作为参数，如果表达式包含Sum就返回True，否则返回False
    :param exp:
    :return:
    """
    if isinstance(exp, Variable):
        return False
    elif isinstance(exp, Number):
        return False
    elif isinstance(exp, Sum):
        return True
    elif isinstance(exp, Product):
        return contains_sum(exp.exp1) or contains_sum(exp.exp2)
    elif isinstance(exp, Power):
        return contains_sum(exp.base) or contains_sum(exp.exponent)
    elif isinstance(exp, Apply):
        return contains_sum(exp.argument)
    else:
        raise TypeError("Not a valid expression.")


# 测试
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

def f(x):
    return (3*x**2 + x) * math.sin(x)


# print("f(5) ={0},\nf_expression.evaluate(x=5) = {1}".format(f(5), f_expression.evaluate(x=5) ))

# 测试 display()
Y = Variable('y')
Z = Variable('z')
A = Variable('a')
B = Variable('b')
# print(Product(Sum(A, B), Sum(Y, Z)))
# print(Product(Sum(A, B), Sum(Y, Z)).expand())
# print(f_expression.expand())

# 测试 _python_expr()和 python_function 效果对标evaluate()功能
test1 = Power(Variable("x"), Number(2))
# print(test1._python_expr())
# print(test1.python_function(x=3))
# print(test1.evaluate(x=3))

"""
eval() 说明:
eval() 函数用来执行一个字符串表达式，并返回表达式的值。
语法:
eval(expression[, globals[, locals]])
参数:
expression -- 表达式。
globals -- 变量作用域，全局命名空间，如果被提供，则必须是一个字典对象。
locals -- 变量作用域，局部命名空间，如果被提供，可以是任何映射对象。
返回值:
eval() 函数将字符串 expression 解析为 Python 表达式，并在指定的命名空间中执行它。
注意点: eval() 函数会执行字符串内部的任何代码，风险点：恶意代码注入.
"""

print(Product(Variable("c"), Variable("x")).derivative(Variable("x")))

