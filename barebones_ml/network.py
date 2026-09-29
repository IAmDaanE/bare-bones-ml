import numpy as np
import pygame
from .losses import Losses

class Network:
    def __init__(self, loss_function_string):
        self.layers = []
        self.num_layers = 0
        self.input_size = 0
        self.output_size = 0
        self.hidden_layer_size = 0
        self.cached_hidden_layer_size = 0
        self.screen = None
        self.font = None
        self.epoch = 0 # should be updated in the training loop, just for visualization
        self.loss = 0 # should be updated in the training loop, just for visualization
        self.current_lr = 0 # should be updated in the training loop, just for visualization
        self.loss_function = Losses.string_map[loss_function_string]

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, prediction, true_value):
        loss_grad_func = Losses.gradient_map[self.loss_function]
        gradient = loss_grad_func(prediction, true_value)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)