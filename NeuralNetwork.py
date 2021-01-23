"""

"""

import numpy, copy

def ReLU(weightedInputs):
    return numpy.append(weightedInputs, [0]).max()

class layerOutput:
    """
    This should act as a data type for the output of a layer, this should carry information such as the output of the neurons some information we may wish to log as well as extra paramters we need to carry over to the next iteration
    """
    def __init__(self, neuronOutputs):
        self.neuronOutputs = neuronOutputs

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