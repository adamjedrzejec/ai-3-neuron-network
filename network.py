import layer
from functionsDerivatives import ActivationFunctionTypes as aft
import numpy as np


class Network:
    layers = []

    def __init__(self, networkModel):
        # for neuronsInLayer in networkModel:
        #   print(neuronsInLayer)
        #   self.layers.append(layer.Layer())

        self.layers.append(layer.Layer(2, 3, aft.LogisticFunction))
        self.layers.append(layer.Layer(3, 5, aft.LogisticFunction))
        self.layers.append(layer.Layer(5, 2, aft.LogisticFunction))

    def train(self, networkInput, predictedOutput, learningRate):
        self.backward(np.subtract(predictedOutput, self.forward(networkInput)))
        self.adjust(learningRate)

    def evaluate(self, networkInput):
        return self.forward(networkInput)

    def forward(self, X):
        for l in self.layers:
            X = l.forward(X)
        return X

    def backward(self, grad):
        for l in self.layers[::-1]:
            grad = l.backward(grad)
        return grad

    def adjust(self, eta):
        for l in self.layers:
            l.adjust(eta)
