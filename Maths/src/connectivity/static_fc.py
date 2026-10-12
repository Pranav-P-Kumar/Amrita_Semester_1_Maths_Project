from pathlib import Path
import numpy as np
import pandas as pd

def load_roi_data(csv_path):
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.startswith("ROI_")]
    if len(cols) < 2: raise ValueError("Need at least two ROI_* columns.")
    X = df[cols].to_numpy(float)
    if X.shape[0] < 3 or not np.isfinite(X).all(): raise ValueError("Invalid ROI data.")
    if np.any(np.std(X, axis=0) == 0): raise ValueError("A ROI is constant.")
    return X, cols, df

def calculate_static_fc(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 2: raise ValueError("X must be timepoints x >=2 ROIs.")
    if not np.isfinite(X).all() or np.any(np.std(X, axis=0) == 0): raise ValueError("Invalid or constant ROI series.")
    C = np.corrcoef(X, rowvar=False)
    C = (C + C.T)/2
    np.fill_diagonal(C, 1.0)
    return C

def save_fc_csv(matrix, roi_names, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=roi_names, columns=roi_names).to_csv(path)
