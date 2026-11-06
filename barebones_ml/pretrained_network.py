from .network import Network
from .pretrained_layer import PreTrainedLayer

import json

class PreTrainedNetwork:
    def __init__(self, model_location):
        self.model_location = model_location
        amount_layers = 0
        activations = []
        with open(self.model_location / "model.json") as f:
            raw_unloaded_data = json.load(f)
            amount_layers = raw_unloaded_data["amount_layers"]
            activations = raw_unloaded_data["activations"]
        self.network = Network("softmax_cross_entropy")
        for i in range(amount_layers):
            self.network.add(PreTrainedLayer(model_location / f"layer_{i}_weights.npy", model_location / f"layer_{i}_biases.npy", activations[i]))

    def forward(self, inputs):
        return self.network.forward(inputs)