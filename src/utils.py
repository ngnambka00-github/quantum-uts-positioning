from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# remove constant column (meaningless)
def filter_constant_column(data_df: pd.DataFrame, threshold: int):
    ret_columns = []

    for col in data_df.columns: 
        if len(pd.unique(data_df[col])) < threshold: 
            ret_columns.append(col)

    return ret_columns

# separate data from data frame
def separate_data(data: pd.DataFrame):
    _x = data.drop(columns=["Pos_x", "Pos_y", "Floor_ID"], axis=1)
    _y = data[["Pos_x", "Pos_y"]]

    _x = _x.to_numpy()
    _y = _y.to_numpy()

    return _x, _y

# normalize data from data frame
def normalize_data(data: pd.DataFrame):
    cell = data.iloc[:, 0:557]
    data.iloc[:, 0:557] = np.where(cell <= 0, (cell + 100)/100, 0)

    return data