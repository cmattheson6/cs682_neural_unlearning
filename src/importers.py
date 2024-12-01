

import pandas as pd
from ucimlrepo import fetch_ucirepo


def import_census_data() -> [pd.DataFrame, pd.DataFrame]:

    # fetch dataset
    result = fetch_ucirepo(id=2)
    X = result.data.features
    y = result.data.targets

    return X, y


# ProPublica COMPAS Data Source
def import_crime_data():

    crime_df = pd.read_csv(
        "https://raw.githubusercontent.com/propublica/compas-analysis/refs/heads/master/compas-scores.csv"
    )
    crime_filtered = crime_df[[
        'sex',
        'age',
        'race',
        'juv_fel_count',
        'decile_score',
        'juv_misd_count',
        'juv_other_count',
        'priors_count',
        'days_b_screening_arrest',
        'c_jail_in',
        'c_jail_out',
        'c_charge_degree',
        'c_charge_desc',
        'is_recid',
        'type_of_assessment',
        'score_text']]

    crime_filtered = crime_filtered[
        (crime_df['days_b_screening_arrest'] <= 30)
        & (crime_df['days_b_screening_arrest'] >= -30)
        & (crime_df['is_recid'] != -1)
        & (crime_df['c_charge_degree'] != "O")
        & (crime_df['score_text'] != 'N/A')
        ]

    X = crime_filtered.drop(['is_recid'], axis=1)
    y = crime_filtered['is_recid']

    return X, y


def import_bank_data():

    result = fetch_ucirepo(id=222)
    X = result.data.features
    y = result.data.targets

    return X, y
