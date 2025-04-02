import numpy as np


class LearningRate:

    def __init__(self, lr_type, start_lr, delay, gamma):
        if lr_type == "ConstantAndDelay":
            self.lr_type = "ConstantAndDelay"
        elif lr_type == "RescaledLinear":
            self.lr_type = "RescaledLinear"
        else:
            assert False

        self.start_lr = start_lr
        self.delay = delay
        self.gamma = gamma


    def get_lr(self, current_iter):
        if self.lr_type == "RescaledLinear":
            return self.start_lr / (1 + (1 - self.gamma) * (current_iter + 1))
        elif self.lr_type == "ConstantAndDelay":
            if current_iter <= self.delay:
                return self.start_lr
            return self.start_lr * 1 / (current_iter - self.start_lr)


