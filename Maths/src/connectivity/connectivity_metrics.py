import numpy as np
import pandas as pd

def summarize_dynamic_fc(matrices, roi_names):
    A = np.asarray(matrices, dtype=float)
    if A.ndim != 3 or A.shape[1] != A.shape[2]: raise ValueError("Expected windows x ROIs x ROIs.")
    upper = np.triu_indices(A.shape[1], 1)
    rows = []
    for k, mat in enumerate(A, 1):
        v = mat[upper]
        rows.append({"window_id": k, "mean_signed_fc": v.mean(),
          "mean_absolute_fc": np.abs(v).mean(),
          "positive_edge_fraction": np.mean(v > 0), "negative_edge_fraction": np.mean(v < 0)})
    edge_rows = []
    for i,j in zip(*upper):
        v = A[:,i,j]
        edge_rows.append({"roi_1": roi_names[i], "roi_2": roi_names[j],
          "mean_fc": v.mean(), "std_fc": v.std(ddof=0), "min_fc": v.min(), "max_fc": v.max()})
    return pd.DataFrame(rows), pd.DataFrame(edge_rows)
