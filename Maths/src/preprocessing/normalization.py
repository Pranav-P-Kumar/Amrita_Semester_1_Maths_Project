"""Per-ROI z-score normalization."""
import pandas as pd

def zscore_normalize(roi_df):
    means = roi_df.mean(axis=0)
    stds = roi_df.std(axis=0, ddof=0)
    if (stds == 0).any():
        raise ValueError("Cannot standardize a constant ROI.")
    normalized = (roi_df - means) / stds
    stats = pd.DataFrame({
        "ROI": roi_df.columns,
        "mean_before": means.values,
        "std_before": stds.values,
        "mean_after": normalized.mean(axis=0).values,
        "std_after": normalized.std(axis=0, ddof=0).values,
    })
    return normalized, stats
