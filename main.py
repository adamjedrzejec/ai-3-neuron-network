import layer
import network
from functionsDerivatives import ActivationFunctionTypes as aft


n = network.Network([1, 3, 2])
for i in range(100):
    n.train([-1, 1], [0, 0], -2)
print(n.evaluate([-1, 1]))
