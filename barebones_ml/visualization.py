import pygame

class Visualization:
    def __init__(self):
        padding = 20
        self.network_rect = pygame.Rect(padding, padding, 500, 420)
        self.graph_rect = pygame.Rect(padding * 2 + 500, padding, 420, 420)
        self.info_rect = pygame.Rect(padding, padding * 2 + 420, 960, 200)
        self.font = pygame.font.Font("../assets/JetBrainsMono-Regular.ttf", 18)

    def draw_ui_shell(self, screen):
        pygame.draw.rect(screen, (0, 173, 181), self.network_rect)
        pygame.draw.rect(screen, (0, 173, 181), self.graph_rect)
        pygame.draw.rect(screen, (0, 173, 181), self.info_rect)

    def draw_network(self, network_class, screen, hor_padding=20, vert_padding=15, method="proportional"):
        window_width = screen.get_width()
        window_height = screen.get_height()
        hidden_amount = len(network_class.layers) - 2
        input_size = network_class.layers[0].n_in
        hidden_size = network_class.layers[1].n_in
        output_size = network_class.layers[-1].n_out
        biggest_node_amount = max(input_size, hidden_size, output_size)
        hor_gap = (window_width - 2 * hor_side_offset) / (hidden_amount + 1)
        if method == "proportional":
            input_node_gap = (window_height - 2 * vert_padding) / (biggest_node_amount + 1)
            hidden_node_gap = input_node_gap
            output_node_gap = input_node_gap
        elif method == "stretched":
            input_node_gap = (window_height - 2 * vert_padding) / (input_size + 1)
            hidden_node_gap = (window_height - 2 * vert_padding) / (hidden_size + 1)
            output_node_gap = (window_height - 2 * vert_padding) / (output_size + 1)
        node_radius = 8
        for i in range(input_size):
            y = (window_height / 2) - (input_node_gap * ((input_size - 1) / 2)) + (i * input_node_gap)
            pygame.draw.circle(screen, (255,255,255), (hor_padding, y), node_radius, 3)
        for q in range(hidden_amount):
            for i in range(hidden_size):
                y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (i * hidden_node_gap)
                pygame.draw.circle(screen, (255,255,255), (hor_padding + hor_gap * (q + 1), y), node_radius, 3)
        for i in range(output_size):
            y = (window_height / 2) - (output_node_gap * ((output_size - 1) / 2)) + (i * output_node_gap)
            pygame.draw.circle(screen, (255,255,255), (hor_padding + (hidden_amount + 1) * hor_gap, y), node_radius, 3)
        for q in range(hidden_amount):
            if q == 0:
                start_x = hor_padding
                end_x = hor_padding + hor_gap
                for i in range(input_size):
                    start_y = (window_height / 2) - (input_node_gap * ((input_size - 1) / 2)) + (i * input_node_gap)
                    for p in range(hidden_size):
                        weight = layers[q].weights[i, p]
                        if weight > 0:
                            color = (255, 255, 255)
                        else:
                            color = (0, 134, 212)
                        end_y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (p * hidden_node_gap)
                        pygame.draw.line(screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))
            else:
                start_x = hor_padding + q * hor_gap
                end_x = hor_padding + (q + 1) * hor_gap
                for i in range(hidden_size):
                    start_y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (i * hidden_node_gap)
                    for p in range(hidden_size):
                        weight = network_class.layers[q].weights[i, p]
                        if weight > 0:
                            color = (255, 255, 255)
                        else:
                            color = (0, 134, 212)
                        end_y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (p * hidden_node_gap)
                        pygame.draw.line(screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))
        for q in range(hidden_size):
            start_x = hor_padding + hor_gap * (hidden_amount)
            end_x = hor_padding + hor_gap * (hidden_amount + 1)
            start_y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (q * hidden_node_gap)
            for p in range(output_size):
                weight = network_class.layers[hidden_amount].weights[q, p]
                if weight > 0:
                    color = (255, 255, 255)
                else:
                    color = (0, 134, 212)
                end_y = (window_height / 2) - (output_node_gap * ((output_size - 1) / 2)) + (p * output_node_gap)
                pygame.draw.line(screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))

    def draw_graph(self, network_class, screen):
        pass

    def draw_info(self, network_class, screen, draw_bools):
        if draw_bools["epoch"]:
            epoch_text = self.font.render(f"epoch: {network_class.epoch}", True, (255, 255, 255))
            screen.blit(epoch_text, (6, 6))
        if draw_bools["loss"]:
            loss_text = self.font.render(f"loss: {network_class.loss}", True, (255, 255, 255))
            screen.blit(loss_text, (6, 100))

    def full_draw(self, screen, network_class):
        network_surface = pygame.Surface((500, 420))
        graph_surface = pygame.Surface((420, 420))
        info_surface = pygame.Surface((960, 200))
        self.draw_network(network_class, network_surface)
        self.draw_graph(network_class, screen)
        draw_bools = {
            "epoch": True,
            "loss": True,
            "learning_rate": True,
            "amount_layers": True,
            "hidden_size": True,
            "model_name": True
        }
        self.draw_info(network_class, screen, draw_bools)
        screen.blit(network_surface, (20, 20))
        screen.blit(graph_surface, (540, 20))
        screen.blit(info_surface, (20, 440))
        self.draw_ui_shell(screen)