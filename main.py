import layer
from layer import ActivationFunctionTypes as aft

a = layer.Layer(2, 3, aft.LogisticFunction)

print(a.forward([-1, 1]))
a.backward([1, 2])

arr = [1.22, 4.44, 12.11]
print(sum(arr))
