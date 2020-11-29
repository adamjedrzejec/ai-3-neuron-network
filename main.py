import layer
import network
from functionsDerivatives import ActivationFunctionTypes as aft
import classifier
import random

EPOCHS = 1000

MODES_PER_CLASSIFIER = 1
SAMPLES_PER_MODE = 5

c1 = classifier.Classifier(MODES_PER_CLASSIFIER, SAMPLES_PER_MODE)
c2 = classifier.Classifier(MODES_PER_CLASSIFIER, SAMPLES_PER_MODE)

pointsFromC1 = c1.getAllPoints()
pointsFromC2 = c2.getAllPoints()

inputOutput1 = list(map(lambda point: [point, [0, 1]], pointsFromC1))
inputOutput2 = list(map(lambda point: [point, [1, 0]], pointsFromC2))

inputsOutputs = inputOutput1 + inputOutput2


n = network.Network([1, 3, 2])
for i in range(EPOCHS):
    random.shuffle(inputsOutputs)
    for io in inputsOutputs:
        n.train(io[0], io[1], 1)

print('expected output:', inputsOutputs[0][1])
print('output:         ', n.evaluate(inputsOutputs[0][0]))
