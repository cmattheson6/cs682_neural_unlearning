
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from copy import deepcopy


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
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        values = self.X.iloc[idx]
        label = self.y.iloc[idx]

        return values, label


def prepare_sampled_datasets(X: pd.DataFrame, y: pd.DataFrame, group_colname: str):

    train_pct = 0.8
    validation_pct = 0.1
    test_pct = 0.1

    idx_filter = StratifiedShuffleSplit(n_splits=1, train_size=1 - test_pct, test_size=test_pct)
    train_val_idxs, test_idxs = idx_filter.split(X, y=[X[group_colname]])[0]
    test_X = X[test_idxs]
    test_y = y[test_idxs]
    test_ds = BaseClassifierDataset(test_X, test_y)

    train_val_X = X[train_val_idxs]
    tran_val_y = y[train_val_idxs]
    idx_filter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_pct/(train_pct+validation_pct),
        test_size=validation_pct/(train_pct+validation_pct)
    )
    train_idxs, val_idxs = idx_filter.split(train_val_X, y=[train_val_X[group_colname]])[0]
    train_X = train_val_X[train_idxs]
    train_y = tran_val_y[train_idxs]
    train_ds = BaseClassifierDataset(train_X, train_y)
    val_X = train_val_X[val_idxs]
    val_y = tran_val_y[val_idxs]
    val_ds = BaseClassifierDataset(val_X, val_y)

    return train_ds, val_ds, test_ds
