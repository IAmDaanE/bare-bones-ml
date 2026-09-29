import numpy as np
from .activations import Activations
from .weight_initializers import WeightInitializers

class Layer:
    def __init__(self, n_in, n_out, activation_string, initializer_string):
        initializer = WeightInitializers.string_map[initializer_string]
        self.weights = initializer(n_in, n_out)
        self.biases = np.zeros((1, n_out))
        self.activation = Activations.string_map[activation_string]
        self.n_in = n_in
        self.n_out = n_out

    def forward(self, inputs):
        self.cached_inputs = inputs
        self.pre_activation = inputs @ self.weights + self.biases
        return self.activation(self.pre_activation)

    def backward(self, incoming_gradient):
        activation_grad_func = Activations.gradient_map[self.activation]
        activation_gradient = activation_grad_func(self.pre_activation)
        gradient_after_activation = incoming_gradient * activation_gradient
        self.weight_gradient = self.cached_inputs.T @ gradient_after_activation
        self.bias_gradient = np.sum(gradient_after_activation, axis=0, keepdims=True)
        return gradient_after_activation @ self.weights.T

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weight_gradient
        self.biases -= learning_rate * self.bias_gradient