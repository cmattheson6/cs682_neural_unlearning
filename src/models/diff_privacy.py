
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
        # TODO: need to figure out how to properly set these somewhere based on each NN
        C = 4
        sigma = 4

        # calculate the norm of the gradient and divide the gradient by the norm
        grad_norm = torch.linalg.norm(grad_output, dim=1) / C
        grad_norm = f.threshold(grad_norm * -1, threshold=-1, value=-1) * -1
        grad_input = (grad_input.t() / grad_norm).t()
        # add noise with normal distribution
        noise = torch.zeros_like(grad_input)
        noise = torch.nn.init.normal_(noise, mean=0, std=(C * sigma) ** 2)
        # divide noise by batch size
        batch_size = grad_output.shape[0]
        noise = noise / batch_size
        grad_input += noise

        return grad_output


def apply_differential_privacy(x):

    result = DifferentiallyPrivateActivation.apply(x)

    return result


class DiffPrivacyFC(nn.Module):

    def __init__(self, input_size, num_classes):

        super().__init__()
        h1_size = 30
        h2_size = 20

        self.fc1 = nn.Linear(input_size, h1_size)
        nn.init.kaiming_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(h1_size, h2_size)
        nn.init.kaiming_normal_(self.fc2.weight)

        self.fc3 = nn.Linear(h2_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):

        # normal three-layer function that adds differential privacy into the backwards pass
        h1 = f.relu(apply_differential_privacy(self.fc1(x)))
        h2 = f.relu(apply_differential_privacy(self.fc2(h1)))
        scores = apply_differential_privacy(self.fc3(h2))

        return scores
