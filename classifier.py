import numpy as np


class Classifier:
    def __init__(self, modesPerClassifier, samplesPerMode):
        self.modes = []
        self.variance = np.random.uniform(0.1, 0.2)

        self.modesPerClassifier = modesPerClassifier
        self.samplesPerMode = samplesPerMode

        self.generateModes()

    def generateModes(self):
        self.modes = []

        for i in range(self.modesPerClassifier):
            meanX = np.random.uniform()
            meanY = np.random.uniform()
            mode = Mode(meanX, meanY)
            mode.generateSamples(self.samplesPerMode, self.variance)
            self.modes.append(mode)

    def getAllSamples(self):
        x = []
        y = []

        for mode in self.modes:
            for sample in mode.samples:
                x.append(sample.x)
                y.append(sample.y)
        return x, y


class Mode:
    def __init__(self, meanX, meanY):
        self.samples = []

        self.meanX = meanX
        self.meanY = meanY

    def generateSamples(self, samplesPerMode, variance):
        self.samples = []

        for i in range(samplesPerMode):
            x = np.random.normal(loc=self.meanX, scale=(variance*variance))
            y = np.random.normal(loc=self.meanY, scale=(variance*variance))
            sample = Sample(x, y)
            self.samples.append(sample)


class Sample:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y
