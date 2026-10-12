from pathlib import Path
import numpy as np
import pandas as pd
from .static_fc import calculate_static_fc

def sliding_window_fc(X, window_size=50, step_size=10, tr=2.0):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2: raise ValueError("X must be 2D: timepoints x ROIs.")
    n, _ = X.shape
    if not np.isfinite(X).all(): raise ValueError("X contains non-finite values.")
    if window_size < 3 or window_size > n: raise ValueError("Invalid window_size.")
    if step_size < 1 or tr <= 0: raise ValueError("step_size and TR must be positive.")
    mats, rows = [], []
    for k, start in enumerate(range(0, n-window_size+1, step_size), 1):
        end = start + window_size
        mats.append(calculate_static_fc(X[start:end]))
        rows.append({"window_id": k, "start_index_0based": start,
          "end_index_0based_exclusive": end, "start_time_seconds": start*tr,
          "end_time_seconds_exclusive": end*tr,
          "center_time_seconds": (start+(window_size-1)/2)*tr,
          "n_timepoints": window_size})
    return np.stack(mats), pd.DataFrame(rows)

def save_dynamic_fc(matrices, metadata, roi_names, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out/"dynamic_fc_matrices.npy", matrices)
    metadata.to_csv(out/"window_metadata.csv", index=False)
    for k, mat in enumerate(matrices, 1):
        pd.DataFrame(mat, index=roi_names, columns=roi_names).to_csv(out/f"window_{k:02d}_fc.csv")
