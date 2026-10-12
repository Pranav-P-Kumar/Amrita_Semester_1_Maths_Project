"""End-to-end preprocessing pipeline for ROI time-series."""
from pathlib import Path
import argparse
import json
import pandas as pd
from scipy.signal import detrend
from .cleaning import clean_roi_data
from .normalization import zscore_normalize
from .filtering import butterworth_bandpass

def preprocess_dataframe(raw_df, tr_seconds=2.0, do_detrend=True,
                         do_filter=True, low_hz=0.01, high_hz=0.10,
                         filter_order=2):
    roi_cols = [c for c in raw_df.columns if str(c).startswith("ROI_")]
    roi = clean_roi_data(raw_df, roi_cols)
    if do_detrend:
        roi = pd.DataFrame(detrend(roi.to_numpy(), axis=0, type="linear"),
                           columns=roi.columns, index=roi.index)
    if do_filter:
        roi = butterworth_bandpass(roi, tr_seconds, low_hz, high_hz, filter_order)
    processed, stats = zscore_normalize(roi)
    output = raw_df.drop(columns=roi_cols).copy()
    for col in roi_cols:
        output[col] = processed[col].values
    output = output[[c for c in raw_df.columns if c in output.columns]]
    report = {
        "n_timepoints": int(len(raw_df)), "n_rois": len(roi_cols),
        "TR_seconds": float(tr_seconds), "sampling_frequency_hz": 1/tr_seconds,
        "Nyquist_hz": 1/(2*tr_seconds), "detrending": do_detrend,
        "bandpass_filtering": do_filter,
        "filter_band_hz": [low_hz, high_hz] if do_filter else None,
        "normalization": "per-ROI z-score",
        "note": "Synthetic demonstration only; real fMRI needs acquisition-specific motion/confound correction."
    }
    return output, stats, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic/synthetic_fmri.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--tr", type=float, default=2.0)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--low", type=float, default=0.01)
    parser.add_argument("--high", type=float, default=0.10)
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    processed, stats, report = preprocess_dataframe(
        raw, args.tr, True, not args.no_filter, args.low, args.high
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out / "synthetic_fmri_preprocessed.csv", index=False)
    stats.to_csv(out / "normalization_statistics.csv", index=False)
    (out / "preprocessing_report.json").write_text(json.dumps(report, indent=4))
    print("Preprocessing completed:", processed.shape)
    print(stats.to_string(index=False))

if __name__ == "__main__":
    main()
