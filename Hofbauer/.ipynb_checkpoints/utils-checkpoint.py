import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_categorical_histograms(df, categorical_columns):
    fig, axs = plt.subplots(len(categorical_columns), 1, figsize=(10, 3 * len(categorical_columns)))

    for i, column in enumerate(categorical_columns):
        value_counts = df[column].value_counts()
        axs[i].bar(value_counts.index, value_counts.values)
        axs[i].set_title(f'Distribution of {column.replace("_", " ").title()}')
        axs[i].set_xlabel(column.replace("_", " ").title())
        axs[i].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


def standardize_numeric(series: pd.Series, use_log: bool = False) -> pd.Series:
    if use_log:
        series = np.log(series)
    return(series - np.mean(series))/np.std(series)


def one_hot_encode_columns(df, categorical_columns):

    for col in categorical_columns:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col, dtype='int', drop_first=True)],axis=1)
    
    return df