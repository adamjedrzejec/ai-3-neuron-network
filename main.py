import layer
import network
from functionsDerivatives import ActivationFunctionTypes as aft
import numpy as np

n = network.Network([1, 3, 2])

networkOutput = n.forward([-1, 1])
predictedOutput = [1, 1]

print(np.subtract(predictedOutput, networkOutput))

n.backward(np.subtract(predictedOutput, networkOutput))
