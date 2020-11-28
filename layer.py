import numpy as np
import functionsDerivatives as fd
from functionsDerivatives import ActivationFunctionTypes as aft


class Layer:
    # done
    def __init__(self, inputWidth, neuronsInLayer, activationFunction):
        print('layer       init')
        self.linear = Linear(inputWidth, neuronsInLayer)
        self.activation = Activation(activationFunction)

    # done
    def forward(self, X):
        print('layer       forward')
        return self.activation.forward(self.linear.forward(X))

    # done
    def backward(self, grad):
        print('layer       backward')
        return self.linear.backward(self.activation.backward(grad))


class Linear:
    # done
    def __init__(self, inputWidth, neuronsInLayer):
        print('linear      init')
        self.X = None
        self.grad = None

        # 2-D array of neuron weights
        self.weights = np.random.rand(neuronsInLayer, inputWidth)

    # done
    def forward(self, X):
        print('linear      forward')
        self.X = X
        return list(map(lambda w: np.dot(X, w), self.weights))

    def backward(self, grad):
        print('linear      backward')
        return list(map(lambda w: np.dot(w, grad), self.weights))

    def adjust(self, eta):
        self.weights = np.add(self.weights, np.dot(
            np.transpose([self.grad]), [self.X]))


class Activation:
    # done
    def __init__(self, activationFunction):
        print('activation  init')
        self.state = None

        if activationFunction == aft.HeaviSideStepFunction:
            self.activationFunction = fd.heaviSideStepFunction
            self.activationDerivative = fd.heaviSideStepFunctionDerivative
            self.theta = 0.01  # np.random.uniform(.05, .1)
        elif activationFunction == aft.LogisticFunction:
            self.activationFunction = fd.logisticFunction
            self.activationDerivative = fd.logisticFunctionDerivative
            self.theta = 1  # np.random.uniform(2, 4)
        elif activationFunction == aft.Sin:
            self.activationFunction = fd.sinh
            self.activationDerivative = fd.sinhDerivative
            self.theta = np.random.uniform(.005, .01)
        elif activationFunction == aft.Tanh:
            self.activationFunction = fd.tanh
            self.activationDerivative = fd.tanhDerivative
            self.theta = np.random.uniform(.5, 1)
        elif activationFunction == aft.Sign:
            self.activationFunction = fd.sign
            self.activationDerivative = fd.signDerivative
            self.theta = np.random.uniform(.03, .06)
        elif activationFunction == aft.ReLu:
            self.activationFunction = fd.reLu
            self.activationDerivative = fd.reLuDerivative
            self.theta = np.random.uniform(.5, 1)
        elif activationFunction == aft.LeakyReLu:
            self.activationFunction = fd.leakyReLu
            self.activationDerivative = fd.leakyReLuDerivative
            self.theta = np.random.uniform(.5, 1)
        else:
            raise ValueError('Activation function not yet supported')

    # done
    def forward(self, state):
        print('activation  forward')
        self.state = state
        return list(map(lambda s: self.activationFunction(s), state))

    # done
    def backward(self, grad):
        print('activation  backward')
        return list(map(lambda s, g: self.activationDerivative(s) * g, self.state, grad))
