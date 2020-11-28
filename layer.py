import numpy as np
import functionsDerivatives as fd
from enum import Enum


class ActivationFunctionTypes(Enum):
    HeaviSideStepFunction = 1
    LogisticFunction = 2
    Sin = 3
    Tanh = 4
    Sign = 5
    ReLu = 6
    LeakyReLu = 7


class Layer:
    # done
    def __init__(self, inputWidth, neuronsInLayer, activationFunction):
        print('layer:init')
        self.linear = Linear(inputWidth, neuronsInLayer)
        self.activation = Activation(activationFunction)

    # done
    def forward(self, X):
        print('layer:forward')
        return self.activation.forward(self.linear.forward(X))

    # done
    def backward(self, grad):
        print('layer:backward')
        return self.linear.backward(self.activation.backward(grad))


class Linear:
    # done
    def __init__(self, inputWidth, neuronsInLayer):
        print('linear:init')
        self.X = None
        self.grad = None

        # 2-D array of neuron weights
        self.weights = np.random.rand(neuronsInLayer, inputWidth)

    # done
    def forward(self, X):
        print('linear:forward')
        self.X = X
        return list(map(lambda w: np.dot(X, w), self.weights))

    def backward(self, grad):
        print('linear:backward', self.X)
        self.grad = grad
        return 1
        # return sum(grad * self.weights)


class Activation:
    # done
    def __init__(self, activationFunction):
        print('activation:init')
        self.state = None

        if activationFunction == ActivationFunctionTypes.HeaviSideStepFunction:
            self.activationFunction = fd.heaviSideStepFunction
            self.activationDerivative = fd.heaviSideStepFunctionDerivative
            self.theta = 0.01  # np.random.uniform(.05, .1)
        elif activationFunction == ActivationFunctionTypes.LogisticFunction:
            self.activationFunction = fd.logisticFunction
            self.activationDerivative = fd.logisticFunctionDerivative
            self.theta = 1  # np.random.uniform(2, 4)
        elif activationFunction == ActivationFunctionTypes.Sin:
            self.activationFunction = fd.sinh
            self.activationDerivative = fd.sinhDerivative
            self.theta = np.random.uniform(.005, .01)
        elif activationFunction == ActivationFunctionTypes.Tanh:
            self.activationFunction = fd.tanh
            self.activationDerivative = fd.tanhDerivative
            self.theta = np.random.uniform(.5, 1)
        elif activationFunction == ActivationFunctionTypes.Sign:
            self.activationFunction = fd.sign
            self.activationDerivative = fd.signDerivative
            self.theta = np.random.uniform(.03, .06)
        elif activationFunction == ActivationFunctionTypes.ReLu:
            self.activationFunction = fd.reLu
            self.activationDerivative = fd.reLuDerivative
            self.theta = np.random.uniform(.5, 1)
        elif activationFunction == ActivationFunctionTypes.LeakyReLu:
            self.activationFunction = fd.leakyReLu
            self.activationDerivative = fd.leakyReLuDerivative
            self.theta = np.random.uniform(.5, 1)
        else:
            raise ValueError('Activation function not yet supported')

    # done
    def forward(self, state):
        print('activation:forward')
        self.state = state
        return list(map(lambda s: self.activationFunction(s), state))

    def backward(self, grad):
        print('activation:backward')
        # return sum of all grads
        return self.activationDerivative(self.state) * grad
