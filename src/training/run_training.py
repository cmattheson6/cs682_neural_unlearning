
import torch
from torch.utils.data import DataLoader, sampler
from torch.nn import Module
import torch.optim as optim
from .prepare_datasets import BaseClassifierDataset
import torch.nn.functional as f


def train(model: Module, dataset: BaseClassifierDataset):
    USE_GPU = True

    dtype = torch.float32  # we will be using float throughout this tutorial

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Constant to control how frequently we print train loss
    print_every = 100

    print('using device:', device)

    # TODO: how do we want to set these?
    hidden_layer_size = 4000
    learning_rate = 1e-2
    model = model(3 * 32 * 32, hidden_layer_size, 10)

    loader_train = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler.SubsetRandomSampler(range(len(dataset)))
    )

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for t, (x, y) in enumerate(loader_train):
        model.train()  # put model to training mode
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.long)

        scores = model(x)
        loss = f.cross_entropy(scores, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % print_every == 0:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))