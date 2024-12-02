
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from copy import deepcopy
from torch import optim
import torch


def onehot_preprocess_data(X: pd.DataFrame) -> pd.DataFrame:
    schema = X.dtypes.reset_index()

    raw_X = deepcopy(X)
    ordinal_colnames = [col[0] for col in schema.values if col[1] == 'object']
    print(f'ordinal colnames: {ordinal_colnames}')
    results = pd.get_dummies(raw_X, columns=ordinal_colnames)

    X_encoded = results

    # clean nulls
    X_encoded = X_encoded.fillna(-1)

    return X_encoded


class BaseClassifierDataset(Dataset):

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X = onehot_preprocess_data(X)
        self.y = y.to_numpy()
        self.shape = self.X.shape

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        values = torch.Tensor(self.X.iloc[idx, :])
        label = self.y[idx]

        return values, label


def prepare_sampled_datasets(X: pd.DataFrame, y: pd.DataFrame, group_colname: str):

    train_pct = 0.8
    validation_pct = 0.1
    test_pct = 0.1

    idx_filter = StratifiedShuffleSplit(n_splits=1, train_size=1 - test_pct, test_size=test_pct)
    train_val_idxs, test_idxs = list(idx_filter.split(X, y=X[group_colname]))[0]
    test_X = X.iloc[test_idxs, :]
    test_y = y.iloc[test_idxs]
    test_ds = BaseClassifierDataset(test_X, test_y)

    train_val_X = X.iloc[train_val_idxs, :]
    tran_val_y = y.iloc[train_val_idxs]
    idx_filter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_pct/(train_pct+validation_pct),
        test_size=validation_pct/(train_pct+validation_pct)
    )
    train_idxs, val_idxs = list(idx_filter.split(train_val_X, y=train_val_X[group_colname]))[0]
    train_X = train_val_X.iloc[train_idxs, :]
    train_y = tran_val_y.iloc[train_idxs]
    train_ds = BaseClassifierDataset(train_X, train_y)
    val_X = train_val_X.iloc[val_idxs, :]
    val_y = tran_val_y.iloc[val_idxs]
    val_ds = BaseClassifierDataset(val_X, val_y)

    return train_ds, val_ds, test_ds


# TODO: remove before submission
if __name__ == '__main__':
    from src.importers import import_crime_data
    from src.models.diff_privacy import DiffPrivacyTwoLayerFC
    from src.training.run_training import train

    crime_X, crime_y = import_crime_data()
    train_ds, val_ds, test_ds = prepare_sampled_datasets(crime_X, crime_y, 'race')

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    model = DiffPrivacyTwoLayerFC(input_size=train_ds.shape[1], hidden_size=30, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    train(
        model=model,
        data_loader=train_loader,
        train_optimizer=optimizer,
        epoch=1,
        epochs=1,
        device='cuda'
    )
