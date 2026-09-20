import numpy as np

class WeightInitializers:
    @staticmethod
    def he(n_in, n_out): # goed voor relu
        return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

    @staticmethod
    def xavier(n_in, n_out): # goed voor tanh of sigmoid
        return np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)

    @staticmethod
    def random_small(n_in, n_out): # algemeen
        return np.random.randn(n_in, n_out) * 0.01

    @staticmethod
    def ones(n_in, n_out):
        return np.ones((n_in, n_out))

    @staticmethod
    def zeros(n_in, n_out):
        return np.zeros((n_in, n_out))