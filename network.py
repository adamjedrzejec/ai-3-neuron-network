import layer
from functionsDerivatives import ActivationFunctionTypes as aft
import numpy as np


class Network:
    __layers = []

    def __init__(self, networkModel):
        # for neuronsInLayer in networkModel:
        #   print(neuronsInLayer)
        #   self.__layers.append(layer.Layer())

        self.__layers.append(layer.Layer(2, 3, aft.LogisticFunction))
        self.__layers.append(layer.Layer(3, 5, aft.LogisticFunction))
        self.__layers.append(layer.Layer(5, 2, aft.LogisticFunction))

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
