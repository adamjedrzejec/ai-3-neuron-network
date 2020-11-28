from enum import Enum
import numpy as np
import functionsDerivatives as fd


class ActivationFunctionTypes(Enum):
    HeaviSideStepFunction = 1
    LogisticFunction = 2
    Sin = 3
    Tanh = 4
    Sign = 5
    ReLu = 6
    LeakyReLu = 7


class Neuron():
    def __init__(self, weights, activationFunction):
        self.weights = weights
        self.deltaWeights = 'undefined'
        self.setFunction(activationFunction)

    def setFunction(self, activationFunction):
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
            print('Function not yet supported')

    def train(self, X, expected):
        bias = self.weights[0]

        state = np.dot(np.transpose(self.weights), [
                       bias, *X]) + bias

        self.deltaWeights = np.dot(np.transpose([self.weights[0], *X]),
                                   self.theta * (expected - self.activationFunction(state)) * self.activationDerivative(state))

    def examine(self, X):
        bias = self.weights[0]

        state = np.dot(np.transpose(self.weights), [
                       bias, *X]) + bias
        return self.activationFunction(state)

    def updateWeights(self):
        if self.deltaWeights != 'undefined':
            self.weights = np.add(self.weights, self.deltaWeights)
            self.deltaWeights = 'undefined'
        return self.weights
