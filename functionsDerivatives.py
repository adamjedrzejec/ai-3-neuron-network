import math


def heaviSideStepFunction(x):
    if x > 0:
        return 1
    elif x < 0:
        return 0
    return 0.5


def heaviSideStepFunctionDerivative(x):
    return 1


def logisticFunction(x):
    return 1 / (1 + math.exp(-x))


def logisticFunctionDerivative(x):
    return logisticFunction(x) * (1 - logisticFunction(x))


def sinh(x):
    return math.sinh(x)


def sinhDerivative(x):
    return math.cosh(x)


def tanh(x):
    return math.tanh(x)


def tanhDerivative(x):
    return 1.0 - tanh(x) ** 2


def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    return 0


def signDerivative(x):
    return 1


def reLu(x):
    if x > 0:
        return x
    return 0


def reLuDerivative(x):
    if x > 0:
        return 1
    return 0


def leakyReLu(x):
    if x > 0:
        return x
    return 0.01 * x


def leakyReLuDerivative(x):
    if x > 0:
        return 1
    return 0.01
