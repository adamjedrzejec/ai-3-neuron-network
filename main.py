import layer
import network
from functionsDerivatives import ActivationFunctionTypes as aft

# a = layer.Layer(2, 3, aft.LogisticFunction)

# print(a.forward([-1, 1]))
# a.backward([1, 2])

# arr = [1.22, 4.44, 12.11]
# print(sum(arr))

n = network.Network([1, 3, 2])

n.forward([-1, 1])
