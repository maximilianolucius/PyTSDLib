import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import ADF


def calculate_ADF(df, cols):
    """
    Calculates ADF statistics for each column in the dataframe except the 'date' column.
    """
    adf_list = []
    for col in cols:
        series = df[col].values
        adf_result = adfuller(series, maxlag=1)
        adf_list.append(adf_result)
    return np.array(adf_list)


def archADF(df, cols):
    """
    Computes the average ADF statistic using the 'arch' package for each series in the dataframe.
    """
    total_stat = 0
    for col in cols:
        series = df[col].values
        adf = ADF(series)
        total_stat += adf.stat
    return total_stat / len(cols)


def process_dataset(root_path, data_path, use_arch=True):
    """
    Loads the dataset, applies ADF or ARCH ADF test, and returns the average metric.
    """
    df = pd.read_csv(os.path.join(root_path, data_path))
    cols = df.columns[1:]  # Exclude the first column, assuming it's the 'date'

    if use_arch:
        return archADF(df, cols)
    else:
        return calculate_ADF(df, cols)


if __name__ == '__main__':
    datasets = {
        "Exchange": ("./dataset/exchange_rate/", "exchange_rate.csv"),
    }

    for dataset_name, (root_path, data_path) in datasets.items():
        ADFmetric = process_dataset(root_path, data_path, use_arch=True)
        print(f"{dataset_name} ADF metric: {ADFmetric}")
