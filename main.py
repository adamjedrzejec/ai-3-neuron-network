import layer
import network
from functionsDerivatives import ActivationFunctionTypes as aft
import classifier
import random


MODES_PER_CLASSIFIER = 2
SAMPLES_PER_MODE = 5

c1 = classifier.Classifier(MODES_PER_CLASSIFIER, SAMPLES_PER_MODE)
c2 = classifier.Classifier(MODES_PER_CLASSIFIER, SAMPLES_PER_MODE)

pointsFromC1 = c1.getAllPoints()
pointsFromC2 = c2.getAllPoints()

inputOutput1 = list(map(lambda point: [point, [0, 1]], pointsFromC1))
inputOutput2 = list(map(lambda point: [point, [1, 0]], pointsFromC2))

inputsOutputs = inputOutput1 + inputOutput2

random.shuffle(inputsOutputs)


n = network.Network([1, 3, 2])
for i in range(100):
    n.train([-1, 1], [0, 0], 2)
print(n.evaluate([-1, 1]))
