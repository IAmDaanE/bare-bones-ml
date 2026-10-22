import numpy as np
from .activations import Activations

class PreTrainedLayer:
    def __init__(self, weights_location, biases_location, activation_string):
        self.weights = np.load(weights_location)
        self.biases = np.load(biases_location)
        self.activation = Activations.string_map[activation_string]

    def forward(self, inputs, reverse=False):
        if not reverse:
            self.pre_activation = inputs @ self.weights + self.biases
        else:
            self.pre_activation = (inputs - self.biases) @ self.weights
        return self.activation(self.pre_activation)