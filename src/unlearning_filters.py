

# imports
import pandas as pd
from typing import List, Union
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.cluster import KMeans
from copy import deepcopy


# TODO: I need to only remove from the positive results as well


class BaseUnlearningFilter:

    def identify_target_indexes(self, X: pd.DataFrame, pct_remove: float, **kwargs) -> List[int]:

        """
        return a list of row numbers that should be filtered out
        """

        raise ValueError(f'Function {self.identify_target_indexes.__name__} needs to be defined in child classes')

    def prepare_dataset(self, X: pd.DataFrame, pct_remove: float, **kwargs) -> pd.DataFrame:

        idxs = self.identify_target_indexes(X, pct_remove=pct_remove, **kwargs)
        X = X.loc[idxs]

        return X


class UniformUnlearning(BaseUnlearningFilter):

    def identify_target_indexes(self, X: pd.DataFrame, pct_remove: float, **kwargs) -> List[int]:

        """
        return a list of row numbers that should be filtered out with no clustering
        """

        idx_filter = ShuffleSplit(n_splits=1, train_size=1 - pct_remove, test_size=pct_remove)
        train_idxs, test_idxs = idx_filter.split(X)[0]

        return train_idxs


class AdversarialUnlearning(BaseUnlearningFilter):

    def identify_target_indexes(self, X: pd.DataFrame, pct_remove: float, **kwargs) -> List[int]:
        """
        return a list of row numbers that should be filtered out
        """

        num_records = X.shape[0]
        group_sizes_to_filter = 0.2  # aggregate percent of the smallest groups to filter
        group_threshold = num_records * group_sizes_to_filter
        colname: str = kwargs.get('colname')
        groups_lst = X.groupby([colname]).count().sort_values('val', ascending=True).reset_index().to_numpy().tolist()

        underrepresented_groups = list()
        underrepresented_count = 0
        for group_name, group_size in groups_lst:
            if underrepresented_count > group_threshold:
                break

            underrepresented_count += group_size
            underrepresented_groups.append(group_name)

        grouped_X = X[X[colname] in underrepresented_groups]
        idx_filter = StratifiedShuffleSplit(n_splits=1, train_size=1-pct_remove, test_size=pct_remove)
        train_idxs, test_idxs = idx_filter.split(grouped_X, y=[grouped_X[colname]])[0]

        return train_idxs

    def prepare_dataset(self, X: pd.DataFrame, pct_remove: float, **kwargs) -> pd.DataFrame:

        assert 'colname' in kwargs.keys()
        idxs = self.identify_target_indexes(X, pct_remove=pct_remove, **kwargs)
        X = X.loc[idxs]

        return X


class ClusteredUnlearning(BaseUnlearningFilter):

    def onehot_preprocess_data(self, X: pd.DataFrame):
        schema = X.dtypes.reset_index()

        raw_X = deepcopy(X)
        ordinal_colnames = [col[0] for col in schema.values if col[1] == 'object']
        print(f'ordinal colnames: {ordinal_colnames}')
        results = pd.get_dummies(raw_X, columns=ordinal_colnames)
        print(results[0:10])

        X_encoded = results

        # clean nulls
        X_encoded = X_encoded.fillna(-1)
        print(X_encoded)

        return X_encoded

    def generate_cluster_labels(self, X_encoded: pd.DataFrame, num_clusters: int) -> pd.DataFrame:
        model: Union[KMeans, object] = KMeans(n_clusters=num_clusters).fit(X_encoded)
        group_labels = model.predict(X_encoded)

        return group_labels

    def append_cluster_group_label(
            self, X: pd.DataFrame, num_clusters: int, group_colname: str = 'group_label'
    ) -> pd.DataFrame:

        X_encoded = self.onehot_preprocess_data(X)

        group_labels = self.generate_cluster_labels(X_encoded, num_clusters=num_clusters)
        X[group_colname] = group_labels

        return X

    def identify_target_indexes(self, X: pd.DataFrame, pct_remove: float, **kwargs) -> List[int]:

        """
        return a list of row numbers that should be filtered out
        """

        num_records = X.shape[0]
        group_sizes_to_filter = 0.2  # aggregate percent of the smallest groups to filter
        group_threshold = num_records * group_sizes_to_filter
        colname: str = kwargs.get('colname')
        groups_lst = X.groupby([colname]).count().sort_values('val', ascending=True).reset_index().to_numpy().tolist()

        underrepresented_groups = list()
        underrepresented_count = 0
        for group_name, group_size in groups_lst:
            if underrepresented_count > group_threshold:
                break

            underrepresented_count += group_size
            underrepresented_groups.append(group_name)

        grouped_X = X[X[colname] in underrepresented_groups]
        idx_filter = StratifiedShuffleSplit(n_splits=1, train_size=1-pct_remove, test_size=pct_remove)
        train_idxs, test_idxs = idx_filter.split(grouped_X, y=[grouped_X[colname]])[0]

        return train_idxs

    def prepare_dataset(self, X: pd.DataFrame, pct_remove: float, **kwargs) -> pd.DataFrame:

        assert 'num_clusters' in kwargs.keys()
        assert 'colname' in kwargs.keys()
        num_clusters = kwargs.get('num_clusters')
        group_colname = 'group_label'
        X = self.append_cluster_group_label(X, group_colname=group_colname, num_clusters=num_clusters)
        idxs = self.identify_target_indexes(X, pct_remove=pct_remove, colname=group_colname, **kwargs)
        X = X.loc[idxs]

        return X






