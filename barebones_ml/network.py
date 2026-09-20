import numpy as np
import pygame
from losses import Losses

class Network:
    def __init__(self, loss_function):
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
        self.loss_function = loss_function

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

    def visualize(self, input_size, hidden_size, hidden_amount, output_size, window_width, window_height, method="stretched"):
        if not self.screen:
            pygame.init()
            self.screen = pygame.display.set_mode((window_width, window_height))
            self.font = pygame.font.Font(None, 32)
        self.screen.fill((0,0,0))
        hor_side_offset = 40
        vert_side_offset = 60
        hor_gap = (window_width - 2 * hor_side_offset) / (hidden_amount + 1)
        biggest_node_amount = max(input_size, hidden_size, output_size)
        if method == "proportional":
            input_node_gap = (window_height - 2 * vert_side_offset) / (biggest_node_amount + 1)
            hidden_node_gap = input_node_gap
            output_node_gap = input_node_gap
        elif method == "stretched":
            input_node_gap = (window_height - 2 * vert_side_offset) / (input_size + 1)
            hidden_node_gap = (window_height - 2 * vert_side_offset) / (hidden_size + 1)
            output_node_gap = (window_height - 2 * vert_side_offset) / (output_size + 1)
        node_radius = 8
        for i in range(input_size):
            y = (window_height / 2) - (input_node_gap * ((input_size - 1) / 2)) + (i * input_node_gap)
            pygame.draw.circle(self.screen, (255,255,255), (hor_side_offset, y), node_radius, 3)
        for q in range(hidden_amount):
            for i in range(hidden_size):
                y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (i * hidden_node_gap)
                pygame.draw.circle(self.screen, (255,255,255), (hor_side_offset + hor_gap * (q + 1), y), node_radius, 3)
        for i in range(output_size):
            y = (window_height / 2) - (output_node_gap * ((output_size - 1) / 2)) + (i * output_node_gap)
            pygame.draw.circle(self.screen, (255,255,255), (hor_side_offset + (hidden_amount + 1) * hor_gap, y), node_radius, 3)
        for q in range(hidden_amount):
            if q == 0:
                start_x = hor_side_offset
                end_x = hor_side_offset + hor_gap
                for i in range(input_size):
                    start_y = (window_height / 2) - (input_node_gap * ((input_size - 1) / 2)) + (i * input_node_gap)
                    for p in range(hidden_size):
                        weight = self.layers[q].weights[i, p]
                        if weight > 0:
                            color = (255, 255, 255)
                        else:
                            color = (0, 134, 212)
                        end_y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (p * hidden_node_gap)
                        pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))
            else:
                start_x = hor_side_offset + q * hor_gap
                end_x = hor_side_offset + (q + 1) * hor_gap
                for i in range(hidden_size):
                    start_y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (i * hidden_node_gap)
                    for p in range(hidden_size):
                        weight = self.layers[q].weights[i, p]
                        if weight > 0:
                            color = (255, 255, 255)
                        else:
                            color = (0, 134, 212)
                        end_y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (p * hidden_node_gap)
                        pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))
        for q in range(hidden_size):
            start_x = hor_side_offset + hor_gap * (hidden_amount)
            end_x = hor_side_offset + hor_gap * (hidden_amount + 1)
            start_y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (q * hidden_node_gap)
            for p in range(output_size):
                weight = self.layers[hidden_amount].weights[q, p]
                if weight > 0:
                    color = (255, 255, 255)
                else:
                    color = (0, 134, 212)
                end_y = (window_height / 2) - (output_node_gap * ((output_size - 1) / 2)) + (p * output_node_gap)
                pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))
        text = self.font.render(f"epoch: {self.epoch} | loss: {self.loss:.6f} | lr: {self.current_lr:.4f}", True, (255, 255, 255))
        self.screen.blit(text, (window_width / 2 - text.get_width() / 2, (vert_side_offset - text.get_height()) / 2))
        pygame.display.update()

    def check_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                return