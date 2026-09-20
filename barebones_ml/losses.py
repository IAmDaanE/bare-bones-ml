import numpy as np

class Losses:
    @staticmethod
    def mse(prediction, true_value):
        return np.mean((prediction - true_value) ** 2)
    
    @staticmethod
    def mse_grad(prediction, true_value):
        return 2 * (prediction - true_value) / prediction.size

    @staticmethod
    def cross_entropy(prediction, true_value):
        prediction = np.clip(prediction, 1e-15, 1.0 - 1.0e-15)
        return -np.sum(true_value * np.log(prediction)) / prediction.shape[0]
    
    @staticmethod
    def cross_entropy_grad(prediction, true_value):
        prediction = np.clip(prediction, 1e-15, 1.0 - 1.0e-15)
        return (-true_value / prediction) / prediction.shape[0]

    @staticmethod
    def softmax_cross_entropy(prediction_logits, true_value):
        m = prediction_logits.shape[0]
        shift_logits = prediction_logits - np.max(prediction_logits, axis=-1, keepdims=True)
        exps = np.exp(shift_logits)
        softmax_output = exps / np.sum(exps, axis=-1, keepdims=True)
        softmax_output = np.clip(softmax_output, 1e-15, 1.0 - 1.0e-15)
        loss = -np.sum(true_value * np.log(softmax_output)) / m
        return loss

    @staticmethod
    def softmax_cross_entropy_grad(prediction_logits, true_value):
        m = prediction_logits.shape[0]
        shift_logits = prediction_logits - np.max(prediction_logits, axis=-1, keepdims=True)
        exps = np.exp(shift_logits)
        softmax_output = exps / np.sum(exps, axis=-1, keepdims=True)
        return (softmax_output - true_value) / m

    gradient_map = {
        softmax_cross_entropy.__func__: softmax_cross_entropy_grad.__func__,
        mse.__func__: mse_grad.__func__,
        cross_entropy.__func__: cross_entropy_grad.__func__
    }