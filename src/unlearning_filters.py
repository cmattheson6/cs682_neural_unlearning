

# imports
import pandas as pd
from typing import List, Union
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.cluster import KMeans
from copy import deepcopy


class BaseUnlearningFilter:

    def identify_target_indexes(self, X: pd.DataFrame, pct_remove: float, **kwargs) -> List[int]:

        """
        return a list of row numbers that should be filtered out
        """

        raise ValueError(f'Function {self.identify_target_indexes.__name__} needs to be defined in child classes')


class UniformUnlearning(BaseUnlearningFilter):

    def identify_target_indexes(self, X: pd.DataFrame, pct_remove: float, **kwargs) -> List[int]:

        """
        return a list of row numbers that should be filtered out with no clustering
        """

        idx_filter = ShuffleSplit(n_splits=1, train_size=1 - pct_remove, test_size=pct_remove)
        train_idxs, test_idxs = list(idx_filter.split(X))[0]

        return train_idxs


class AdversarialUnlearning(BaseUnlearningFilter):

    def identify_target_indexes(self, X: pd.DataFrame, pct_remove: float, **kwargs) -> List[int]:
        """
        return a list of row numbers that should be filtered out
        """

        assert 'colname' in kwargs.keys()
        num_records = X.shape[0]
        group_sizes_to_filter = 0.2  # aggregate percent of the smallest groups to filter
        group_threshold = num_records * group_sizes_to_filter
        colname: str = kwargs.get('colname')
        groups_lst = X.groupby(colname).size().sort_values(ascending=False)

        underrepresented_groups = list(groups_lst.index)
        underrepresented_count = X.shape[0]
        for i in range(len(underrepresented_groups)):
            if underrepresented_count > group_threshold:
                underrepresented_groups.pop(0)
                underrepresented_count -= groups_lst.values[i]

        grouped_X = X[X[colname].isin(underrepresented_groups)]
        other_X = X[~(X[colname].isin(underrepresented_groups))]
        idx_filter = StratifiedShuffleSplit(n_splits=1, train_size=1-pct_remove, test_size=pct_remove)
        train_idxs, test_idxs = list(idx_filter.split(grouped_X, y=grouped_X[colname]))[0]

        return list(train_idxs) + list(other_X.index)


class ClusteredUnlearning(BaseUnlearningFilter):

    def onehot_preprocess_data(self, X: pd.DataFrame):
        schema = X.dtypes.reset_index()

        raw_X = deepcopy(X)
        ordinal_colnames = [col[0] for col in schema.values if col[1] == 'object']
        print(f'ordinal colnames: {ordinal_colnames}')
        results = pd.get_dummies(raw_X, columns=ordinal_colnames)

        X_encoded = results

        # clean nulls
        X_encoded = X_encoded.fillna(-1)

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

        assert 'colname' in kwargs.keys()
        num_clusters = kwargs.get('num_clusters', 5)
        group_colname = 'group_label'
        X = self.append_cluster_group_label(X, group_colname=group_colname, num_clusters=num_clusters)

        num_records = X.shape[0]
        group_sizes_to_filter = 0.2  # aggregate percent of the smallest groups to filter
        group_threshold = num_records * group_sizes_to_filter
        colname: str = kwargs.get('colname')
        groups_lst = X.groupby(colname).size().sort_values(ascending=False)

        underrepresented_groups = list(groups_lst.index)
        underrepresented_count = X.shape[0]
        for i in range(len(underrepresented_groups)):
            if underrepresented_count > group_threshold:
                underrepresented_groups.pop(0)
                underrepresented_count -= groups_lst.values[i]

        grouped_X = X[X[colname].isin(underrepresented_groups)]
        other_X = X[~(X[colname].isin(underrepresented_groups))]
        idx_filter = StratifiedShuffleSplit(n_splits=1, train_size=1-pct_remove, test_size=pct_remove)
        train_idxs, test_idxs = list(idx_filter.split(grouped_X, y=grouped_X[colname]))[0]

        return list(train_idxs) + list(other_X.index)
