import pandas as pd
# from functional import seq
import re
import numpy as np

def multi_add(df, arr, name):
    for c in range(arr.shape[1]):
        df[f'{name}{c}'] = arr[:, c]

def multi_get(df, name):
    cols = df.keys() if df.__class__ == pd.Series else df.columns
    num_cols = len(list(filter(lambda x: re.match(f'{name}\d+', x),cols)))
    sc = df[[f'{name}{c}' for c in range(num_cols)]]
    return np.array(tuple(sc) if df.__class__ == pd.Series else sc)

