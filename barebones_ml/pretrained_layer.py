import numpy as np

class PreTrainedLayer:
    def __init__(self, weights_location, biases_location, activation):
        self.weights = np.load(weights_location)
        self.biases = np.load(biases_location)
        self.activation = activation

    def forward(self, inputs):
        self.pre_activation = inputs @ self.weights + self.biases
        return self.activation(self.pre_activation)