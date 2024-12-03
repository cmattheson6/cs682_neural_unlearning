
import torch
from tqdm import tqdm
from src.unlearning_filters import *
from src.importers import *
from src.models.diff_privacy import DiffPrivacyFC
from src.fairness_metrcs import find_max_disparate_impact
from src.training.prepare_datasets import prepare_sampled_datasets
from torch.utils.data import DataLoader
import pandas as pd
from torch import optim
import sys
import os


def run_epoch(model, data_loader, train_optimizer, is_train: bool, epoch: int, device: str = 'cuda'):

    loss_criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0
    total_num = 0
    total_correct = 0
    data_bar = tqdm(data_loader)

    unlabelled_data = list()
    predicted_values = pd.Series()
    for data, target in data_bar:
        data, target = data.to(device), target.to(device)
        unlabelled_data.append(data)
        out = model(data)
        predicted = torch.argmax(out, dim=1)
        predicted_values = pd.concat([predicted_values, pd.Series(predicted.cpu().numpy())])
        loss = loss_criterion(out, target)
        total_correct += (predicted == target).sum().item()
        total_num += data.size(0)
        total_loss += loss.item() * data.size(0)

        if is_train:
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

        data_bar.set_description(f'Epoch {epoch} loss: {total_loss / total_num} accuracy: {total_correct / total_num}')

    return torch.vstack(unlabelled_data).cpu(), predicted_values


def train(model, data_loader, train_optimizer, num_epochs, device='cuda'):

    is_train = train_optimizer is not None
    model.train() if is_train else model.eval()
    model.to(device=device)
    # loss_criterion = torch.nn.CrossEntropyLoss()

    # data_bar = tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for epoch in range(num_epochs):
            print(f'Running epoch {epoch}')
            unlabelled_data, predicted_values = run_epoch(
                model=model, data_loader=data_loader, train_optimizer=train_optimizer,
                is_train=is_train, device=device, epoch=epoch
            )

    return unlabelled_data, predicted_values


def run_model_permutation(X: pd.DataFrame, y: pd.DataFrame, pct: float, m: BaseUnlearningFilter, colname: str, num_clusters: int):

    idxs = m.identify_target_indexes(X, pct_remove=pct, colname=colname, num_clusters=num_clusters)
    X = X.iloc[idxs, :]
    y = y.iloc[idxs]
    train_ds, val_ds = prepare_sampled_datasets(X, y)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    model = DiffPrivacyFC(input_size=train_ds.shape[1], num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-6)
    train(
        model=model,
        data_loader=train_loader,
        train_optimizer=optimizer,
        num_epochs=1,
        device='cuda'
    )
    # eval run
    unlabelled_data, predicted_values = train(
        model=model,
        data_loader=val_loader,
        train_optimizer=None,
        num_epochs=1,
        device='cuda'
    )
    val_data = pd.DataFrame(unlabelled_data.numpy(), columns=train_ds.X.columns)
    disp_impact = find_max_disparate_impact(val_data, predicted_values, colname='race')
    return disp_impact


def run_crime_permutations():

    unlearning_pcts = [0.01, 0.05, 0.1, 0.15, 0.2]
    unlearning_methods = [UniformUnlearning, AdversarialUnlearning, ClusteredUnlearning]
    #  across all 3 sources
    crime_X, crime_y = import_crime_data()

    # crime data
    for pct in unlearning_pcts:
        for m in unlearning_methods:
            disp_impact = run_model_permutation(X=crime_X, y=crime_y, pct=pct, m=m(), colname='race', num_clusters=7)

            results = pd.DataFrame([{
                'unlearning_pct': pct,
                'unlearning_method': m.__name__,
                'disparate_impact': disp_impact
            }])
            results.to_csv(os.path.dirname(__file__) + '/results/compas_results.csv', sep=',', header=True, mode='a')


if __name__ == '__main__':
    run_crime_permutations()
