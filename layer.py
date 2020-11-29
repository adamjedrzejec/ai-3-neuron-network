import numpy as np
import functionsDerivatives as fd
from functionsDerivatives import ActivationFunctionTypes as aft


class Layer:
    def __init__(self, inputWidth, neuronsInLayer, activationFunction):
        self.linear = Linear(inputWidth, neuronsInLayer)
        self.activation = Activation(activationFunction)

    def forward(self, X):
        return self.activation.forward(self.linear.forward(X))

    def backward(self, grad):
        return self.linear.backward(self.activation.backward(grad))

    def adjust(self, learningRate):
        self.linear.adjust(learningRate)


class Linear:
    def __init__(self, inputWidth, neuronsInLayer):
        self.X = None
        self.grad = None

        # 2-D array of neuron weights
        self.weights = np.random.rand(neuronsInLayer, inputWidth)

    def forward(self, X):
        self.X = X
        return list(map(lambda w: np.dot(X, w), self.weights))

    def backward(self, grad):
        self.grad = grad
        return np.dot(grad, self.weights)

    def adjust(self, learningRate):
        self.weights = np.add(self.weights, np.dot(
            np.transpose([self.grad]), [self.X]))


class Activation:
    def __init__(self, activationFunction):
        self.state = None

        if activationFunction == aft.HeaviSideStepFunction:
            self.activationFunction = fd.heaviSideStepFunction
            self.activationDerivative = fd.heaviSideStepFunctionDerivative
        elif activationFunction == aft.LogisticFunction:
            self.activationFunction = fd.logisticFunction
            self.activationDerivative = fd.logisticFunctionDerivative
        elif activationFunction == aft.Sin:
            self.activationFunction = fd.sinh
            self.activationDerivative = fd.sinhDerivative
        elif activationFunction == aft.Tanh:
            self.activationFunction = fd.tanh
            self.activationDerivative = fd.tanhDerivative
        elif activationFunction == aft.Sign:
            self.activationFunction = fd.sign
            self.activationDerivative = fd.signDerivative
        elif activationFunction == aft.ReLu:
            self.activationFunction = fd.reLu
            self.activationDerivative = fd.reLuDerivative
        elif activationFunction == aft.LeakyReLu:
            self.activationFunction = fd.leakyReLu
            self.activationDerivative = fd.leakyReLuDerivative
        else:
            raise ValueError('Activation function not yet supported')

    def forward(self, state):
        self.state = state
        return list(map(lambda s: self.activationFunction(s), state))

    def backward(self, grad):
        return list(map(lambda s, g: self.activationDerivative(s) * g, self.state, grad))
