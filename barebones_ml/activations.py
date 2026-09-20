import numpy as np

class Activations:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(float)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_grad(x):
        return np.ones_like(x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_grad(x):
        s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return s * (1.0 - s)

    @staticmethod
    def tanh(x):
        clipped_x = np.clip(x, -500, 500)
        return np.tanh(clipped_x)

    @staticmethod
    def tanh_grad(x):
        clipped_x = np.clip(x, -500, 500)
        t = np.tanh(clipped_x)
        return 1.0 - t ** 2

    gradient_map = {
        relu.__func__: relu_grad.__func__,
        linear.__func__: linear_grad.__func__,
        sigmoid.__func__: sigmoid_grad.__func__,
        tanh.__func__: tanh_grad.__func__
    }