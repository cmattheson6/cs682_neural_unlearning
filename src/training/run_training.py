
import torch
from torch.utils.data import DataLoader, sampler
from torch.nn import Module
import torch.optim as optim
from .prepare_datasets import BaseClassifierDataset
import torch.nn.functional as f
from tqdm import tqdm


def train(model, data_loader, train_optimizer, epoch, epochs, device='cuda'):
    is_train = train_optimizer is not None
    model.train() if is_train else model.eval()
    model.to(device=device)
    loss_criterion = torch.nn.CrossEntropyLoss()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            # total_num += data.size(0)
            # total_loss += loss.item() * data.size(0)
    #         prediction = torch.argsort(out, dim=-1, descending=True)
    #         total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
    #         total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
    #
    #         data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
    #                                  .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
    #                                          total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))
    #
    # return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100

