import layer
from enum import Enum
from layer import ActivationFunctionTypes as aft


class Network:
    layers = []

    def __init__(self, networkModel):
        # for neuronsInLayer in networkModel:
        #   print(neuronsInLayer)
        #   self.layers.append(layer.Layer())

        self.layers.append(layer.Layer(2, 3, aft.LogisticFunction))
        self.layers.append(layer.Layer(3, 2, aft.LogisticFunction))

    def forward(self, X):
        for l in self.layers:
            X = l.forward(X)

    def backward(self, grad):
        for l in self.layers[::-1]:
            grad = l.forward(grad)
