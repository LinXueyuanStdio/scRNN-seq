import numpy as np
import pandas as pd
import scanpy.api as sc


def to_img(x):
    x = x.view(x.size(0), 1, 100, 50)
    return x


def normalization(express_data):
    adata = sc.AnnData(express_data.T.values)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    return pd.DataFrame(adata.X.T, columns=express_data.columns.tolist(), index=express_data.index.tolist())


def norm(x, reverse=False):
    if reverse:
        y = np.power(10, x) - 1.01
        y = np.around(y).astype(np.int32)
        return y
    else:
        return np.log10(x + 1.01)


def minmax_0_to_1(x, reverse=False, minmax=1):
    if reverse:
        # x -> [0, 1]
        return x * minmax
        # minmax_x -> [0, 6]
    else:
        # norm_x -> [0, 6]
        return x / minmax
        # minmax_x -> [0, 1]
