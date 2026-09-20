import numpy as np

class LRDecays:
    @staticmethod
    def exponential_decay(current_lr, factor):
        return factor * current_lr

    @staticmethod
    def step_decay(current_lr, lr_drop, epoch, epoch_interval):
        if epoch % epoch_interval == 0:
            return current_lr - lr_drop
        else:
            return current_lr

    @staticmethod
    def linear_decay(current_lr, lr_drop, lr_min):
        return max(lr_min, current_lr - lr_drop)

    @staticmethod
    def cosine_decay(current_epoch, total_epochs, start_lr, min_lr):
        current_epoch = min(current_epoch, total_epochs)
        progress = current_epoch / total_epochs
        cosine_out = 0.5 * (1.0 + np.cos(np.pi * progress))
        lr = min_lr + (start_lr - min_lr) * cosine_out
        return lr

    @staticmethod
    def SGDR(current_epoch, start_cycle_epochs, cycle_multiplier, start_lr, min_lr):
        cycle_epochs = start_cycle_epochs
        epoch_in_cycle = current_epoch
        while epoch_in_cycle >= cycle_epochs:
            epoch_in_cycle -= cycle_epochs
            cycle_epochs *= cycle_multiplier
        progress = epoch_in_cycle / cycle_epochs
        cosine_out = 0.5 * (1.0 + np.cos(np.pi * progress))
        lr = min_lr + (start_lr - min_lr) * cosine_out
        return lr

    @staticmethod
    def inverse_time_decay(initial_lr, decay_rate, current_epoch, min_lr):
        return max(min_lr, initial_lr / (1.0 + decay_rate * current_epoch))