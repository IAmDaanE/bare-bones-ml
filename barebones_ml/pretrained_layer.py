import numpy as np
from .activations import Activations

class PreTrainedLayer:
    def __init__(self, weights_location, biases_location, activation_string):
        self.weights = np.load(weights_location)
        self.biases = np.load(biases_location)
        self.activation = Activations.string_map[activation_string]

    def forward(self, inputs):
        self.pre_activation = inputs @ self.weights + self.biases
        return self.activation(self.pre_activation)