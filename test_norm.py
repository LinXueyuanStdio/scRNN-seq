import numpy as np
import scanpy.api as sc
import pandas as pd

def norm(x, reverse=False):
    if reverse:
        y = np.power(10, x) - 1.01
        y = np.around(y).astype(np.int32)
        return y
    else:
        return np.log10(x + 1.01)


array = [0, 1, 2, 3, 150, 134]
x = np.asarray(array)
print(norm(x))
print(norm(norm(x), True))
print("-----------------")


def normalization(express_data):
    adata = sc.AnnData(express_data.T.values)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    return pd.DataFrame(adata.X.T, columns=express_data.columns.tolist(), index=express_data.index.tolist())

a = pd.read_csv("./data/true_counts_simulated_dataset1_dropout0.05.csv")

print(normalization(a))
print(normalization(a).describe())
