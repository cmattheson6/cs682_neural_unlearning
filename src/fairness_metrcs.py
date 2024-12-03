

import pandas as pd
from sklearn.metrics import confusion_matrix
import re


def calculate_binary_confusion_matrix(y_actuals: pd.DataFrame, y_predicted: pd.DataFrame):

    tn, fp, fn, tp = confusion_matrix(y_actuals, y_predicted).ravel()
    confusion_matrix_dct = {
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn
    }

    return confusion_matrix_dct


def calculate_demographic_parity():

    pass


def calculate_disparate_impact(X: pd.DataFrame, y_predicted: pd.Series, colname: str):

    conf_matrix = {
        'default_positive': len(X[(X[colname] != 1) & (y_predicted == 1)]),
        'default_negative': len(X[(X[colname] != 1) & (y_predicted == 0)]),
        'minority_positive': len(X[(X[colname] == 1) & (y_predicted == 1)]),
        'minority_negative': len(X[(X[colname] == 1) & (y_predicted == 0)])
    }

    # sensitivity = d / b + d
    sensitivity = conf_matrix.get('minority_positive') / (
            conf_matrix.get('minority_positive') + conf_matrix.get('default_positive')
    )
    # specificity = a / a + c
    specificity = conf_matrix.get('minority_negative') / (
            conf_matrix.get('minority_negative') + conf_matrix.get('default_negative')
    )
    # DI = 1 - specificity / sensitivity
    if sensitivity == 0 or specificity == 0:
        return 0

    disp_impact = (1 - specificity) / sensitivity
    return disp_impact

def find_max_disparate_impact(X: pd.DataFrame, y_predicted: pd.Series, colname: str):
    """
    run above disparate function across all groups and find max disparate impact
    """
    onehot_colnames = [c for c in X.columns if re.match(f'^{colname}.*', c)]
    disp_impacts_lst = list()
    for g in onehot_colnames:
        disp_impacts_lst.append(calculate_disparate_impact(X, y_predicted, colname=g))

    disp_impact = max(disp_impacts_lst)

    return disp_impact


def calculate_equalized_odds():

    pass
