from pathlib import Path
import json
import numpy as np
from src.connectivity.static_fc import load_roi_data, calculate_static_fc, save_fc_csv
from src.connectivity.dynamic_fc import sliding_window_fc, save_dynamic_fc
from src.connectivity.connectivity_metrics import summarize_dynamic_fc

ROOT = Path(__file__).resolve().parent
INPUT = ROOT/"data/processed/synthetic_fmri_preprocessed.csv"
OUT = ROOT/"outputs/connectivity_matrices"

def main():
    X, names, _ = load_roi_data(INPUT)
    static = calculate_static_fc(X)
    OUT.mkdir(parents=True, exist_ok=True)
    save_fc_csv(static, names, OUT/"static_fc.csv")
    np.save(OUT/"static_fc.npy", static)
    dynamic, metadata = sliding_window_fc(X, window_size=50, step_size=10, tr=2.0)
    save_dynamic_fc(dynamic, metadata, names, OUT/"dynamic")
    win, edges = summarize_dynamic_fc(dynamic, names)
    win.to_csv(OUT/"window_fc_summary.csv", index=False)
    edges.to_csv(OUT/"edge_variability_summary.csv", index=False)
    mean_dynamic = dynamic.mean(axis=0)
    np.save(OUT/"mean_dynamic_fc.npy", mean_dynamic)
    report = {"n_timepoints": int(X.shape[0]), "n_rois": int(X.shape[1]),
      "TR_seconds": 2.0, "window_size": 50, "step_size": 10,
      "n_windows": int(len(dynamic)), "dynamic_fc_shape": list(dynamic.shape),
      "mean_absolute_difference_static_vs_mean_dynamic": float(np.mean(np.abs(static-mean_dynamic))),
      "data_note": "Synthetic demonstration data, not measured human fMRI."}
    (OUT/"phase3_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
