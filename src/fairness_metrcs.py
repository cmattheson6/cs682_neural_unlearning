

import pandas as pd
from sklearn.metrics import confusion_matrix


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


def calculate_disparate_impact(x: pd.Series, y_predicted: pd.Series, minority_value):

    conf_matrix = {
        'default_positive': y_predicted[(x != minority_value) & (y_predicted == 1)],
        'default_negative': y_predicted[(x != minority_value) & (y_predicted == 0)],
        'minority_positive': y_predicted[(x == minority_value) & (y_predicted == 1)],
        'minority_negative': y_predicted[(x == minority_value) & (y_predicted == 0)]
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
    disp_impact = (1 - specificity) / sensitivity

    return disp_impact



def calculate_equalized_odds():

    pass