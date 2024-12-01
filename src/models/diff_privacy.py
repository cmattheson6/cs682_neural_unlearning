
import torch.nn.functional as f
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np


class DifferentiallyPrivateActivation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        """
        Don't change anything in the input; just pass it along
        """

        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):

        """
        Goal: decrease the model's reliance on any single point during gradient descent
        1) divide the gradient of each linear layer by its norm
        2) add noise with a defined distribution
        """

        grad_input = grad_output.clone()
        C = 1
        sigma = 1

        # calculate the norm of the gradient and divide the gradient by the norm
        grad_norm = torch.linalg.norm(grad_output, dim=1) / C
        grad_norm = f.threshold(grad_norm * -1, threshold=-1, value=-1) * -1
        grad_input = grad_input / grad_norm
        # add noise with normal distribution
        noise = torch.zeros_like(grad_input)
        noise = torch.nn.init.normal_(noise, mean=0, std=(C * sigma) ** 2)
        # divide noise by batch size
        batch_size = grad_output.shape[0]
        noise = noise / batch_size
        grad_input += noise

        return grad_input


def apply_differential_privacy(x):

    result = DifferentiallyPrivateActivation.apply(x)

    return result


class TwoLayerFC(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):

        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        nn.init.kaiming_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):

        # normal two-layer function that adds differential privacy into the backwards pass
        h1 = f.relu(apply_differential_privacy(self.fc1(x)))
        scores = apply_differential_privacy(self.fc2(h1))

        return scores
