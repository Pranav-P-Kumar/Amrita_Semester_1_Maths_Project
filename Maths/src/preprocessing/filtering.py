"""Zero-phase Butterworth band-pass filtering."""
import pandas as pd
from scipy.signal import butter, sosfiltfilt

def butterworth_bandpass(roi_df, tr_seconds, low_hz=0.01, high_hz=0.10, order=2):
    if tr_seconds <= 0:
        raise ValueError("TR must be positive.")
    fs = 1.0 / tr_seconds
    nyquist = fs / 2.0
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(f"Require 0 < low < high < Nyquist ({nyquist:.4f} Hz).")
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    values = [sosfiltfilt(sos, roi_df[c].to_numpy(dtype=float)) for c in roi_df.columns]
    return pd.DataFrame(dict(zip(roi_df.columns, values)), index=roi_df.index)
