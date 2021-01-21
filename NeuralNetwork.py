"""

"""

import numpy, copy

def ReLU(weightedInputs):
    return numpy.append(weightedInputs, [0]).max()

def neuralNetwork(initialInputs, weights, activationFunction, activationFunctionArguments):
    neuronOutputs = []
    inputs = copy.deepcopy(initialInputs)
    for layerWeights in weights:
        outputs = []
        biasedInputs = numpy.vstack((inputs.transpose(), numpy.array([1 for x in inputs]))).transpose()
        layerWeightedInputs = biasedInputs * layerWeights
        for weightedInputs in  layerWeightedInputs:
            outputs.append(activationFunction(weightedInputs, *activationFunctionArguments))
        outputs = numpy.array(outputs)
        neuronOutputs.append(outputs)
        inputs = copy.deepcopy(outputs)
    return neuronOutputs