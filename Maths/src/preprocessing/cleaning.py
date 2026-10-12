"""Validation and missing-value handling for ROI time-series."""
import numpy as np
import pandas as pd

def clean_roi_data(df, roi_columns=None):
    if roi_columns is None:
        roi_columns = [c for c in df.columns if str(c).startswith("ROI_")]
    if not roi_columns:
        raise ValueError("No ROI columns found (expected ROI_01, ROI_02, ...).")
    missing = [c for c in roi_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing ROI columns: {missing}")
    roi = df[roi_columns].apply(pd.to_numeric, errors="coerce").copy()
    empty = roi.columns[roi.isna().all()].tolist()
    if empty:
        raise ValueError(f"Entirely missing ROI columns: {empty}")
    roi = roi.interpolate(method="linear", axis=0, limit_direction="both")
    if roi.isna().any().any() or not np.isfinite(roi.to_numpy()).all():
        raise ValueError("Missing or non-finite values remain after cleaning.")
    constant = [c for c in roi if roi[c].std(ddof=0) < 1e-12]
    if constant:
        raise ValueError(f"Constant ROI columns: {constant}")
    return roi
