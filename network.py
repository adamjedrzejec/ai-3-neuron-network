import layer
from functionsDerivatives import ActivationFunctionTypes as aft
import numpy as np


class Network:
    def __init__(self, networkModel):
        self.__layers = []
        self.__layers.append(layer.Layer(2, 3, aft.LogisticFunction))
        self.__layers.append(layer.Layer(3, 5, aft.LogisticFunction))
        self.__layers.append(layer.Layer(5, 1, aft.LogisticFunction))

        print(len(self.__layers))

    def train(self, networkInput, predictedOutput, learningRate):
        self.__backward(np.subtract(predictedOutput,
                                    self.__forward(networkInput)))
        self.__adjust(learningRate)

    def evaluate(self, networkInput):
        return self.__forward(networkInput)

    def __forward(self, X):
        for l in self.__layers:
            X = l.forward(X)
        return X

    def __backward(self, grad):
        for l in self.__layers[::-1]:
            grad = l.backward(grad)
        return grad

    def __adjust(self, eta):
        for l in self.__layers:
            l.adjust(eta)
