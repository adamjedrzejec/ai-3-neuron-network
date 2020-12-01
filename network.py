import layer
import numpy as np


class Network:
    def __init__(self, networkModel, activationFunction):
        self.__layers = []

        numberOfInputs = 2
        numberOfOutputs = 2

        for neuronsInLayer in networkModel:
            self.__layers.append(layer.Layer(
                numberOfInputs, neuronsInLayer, activationFunction))
            numberOfInputs = neuronsInLayer

        self.__layers.append(layer.Layer(
            numberOfInputs, numberOfOutputs, activationFunction))

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
