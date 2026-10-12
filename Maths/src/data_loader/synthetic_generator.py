"""
Synthetic fMRI BOLD time-series generator for the Brain Connectome Analysis project.

Creates:
  1. synthetic_fmri.csv       : timepoint + ROI signals (300 rows by default)
  2. synthetic_metadata.csv   : acquisition and generation metadata
  3. synthetic_ground_truth.csv: simulated network-state labels (for validation only)

The data are simulated, not measured from a human participant and not suitable
for clinical interpretation.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


def make_state_loadings(n_rois: int, state: int) -> np.ndarray:
    """Build ROI-to-latent-factor loadings for three different network states."""
    n_modules = 2
    module = np.arange(n_rois) % n_modules
    loadings = np.zeros((n_rois, 1 + n_modules), dtype=float)

    if state == 0:
        # State 1: stronger within-module coupling.
        loadings[:, 0] = 0.18  # weak global factor
        for roi in range(n_rois):
            loadings[roi, 1 + module[roi]] = 0.78
    elif state == 1:
        # State 2: stronger cross-module/global coordination.
        loadings[:, 0] = 0.62
        for roi in range(n_rois):
            loadings[roi, 1 + module[roi]] = 0.35
    else:
        # State 3: relatively distributed, weaker modular organization.
        loadings[:, 0] = 0.42
        for roi in range(n_rois):
            loadings[roi, 1 + module[roi]] = 0.48

    return loadings


def simulate_ar_latents(rng, n_samples, covariance, ar=0.78):
    """Generate multivariate AR(1) latent signals with correlated innovations."""
    n_factors = covariance.shape[0]
    # Small diagonal term ensures numerical stability for Cholesky.
    chol = np.linalg.cholesky(covariance + 1e-8 * np.eye(n_factors))
    innovations = rng.normal(size=(n_samples, n_factors)) @ chol.T
    latent = np.zeros_like(innovations)
    for t in range(1, n_samples):
        latent[t] = ar * latent[t - 1] + np.sqrt(1 - ar**2) * innovations[t]
    return latent


def generate_synthetic_fmri(
    n_timepoints=300,
    n_rois=10,
    tr_seconds=2.0,
    random_seed=42,
    state_boundaries=(100, 200),
    baseline=100.0,
    signal_scale=1.5,
):
    """
    Return (signals_df, metadata_df, ground_truth_df).

    Each ROI signal is a combination of state-dependent correlated latent
    activity, ROI-specific autocorrelated noise, and a small slow drift.
    """
    if n_timepoints < 30:
        raise ValueError("Use at least 30 time points for meaningful window analysis.")
    if n_rois < 4:
        raise ValueError("Use at least 4 ROIs.")
    b1, b2 = state_boundaries
    if not (0 < b1 < b2 < n_timepoints):
        raise ValueError("state_boundaries must satisfy 0 < b1 < b2 < n_timepoints.")

    rng = np.random.default_rng(random_seed)
    states = np.zeros(n_timepoints, dtype=int)
    states[b1:b2] = 1
    states[b2:] = 2

    # Three latent factors: one global and two module-related factors.
    n_factors = 3
    latent_by_state = {}
    for state in range(3):
        # Mild correlation between latent factors.
        latent_cov = np.array([
            [1.00, 0.12, 0.12],
            [0.12, 1.00, 0.08],
            [0.12, 0.08, 1.00],
        ])
        latent_by_state[state] = simulate_ar_latents(
            rng, n_timepoints, latent_cov, ar=0.78
        )

    roi_signals = np.zeros((n_timepoints, n_rois), dtype=float)
    roi_ar = rng.normal(size=(n_timepoints, n_rois))
    for roi in range(n_rois):
        for t in range(1, n_timepoints):
            roi_ar[t, roi] = 0.55 * roi_ar[t - 1, roi] + np.sqrt(1 - 0.55**2) * roi_ar[t, roi]

    for state in range(3):
        idx = np.where(states == state)[0]
        loadings = make_state_loadings(n_rois, state)
        latent = latent_by_state[state]
        # Each ROI combines global + module factors; residual keeps variance finite.
        component = latent @ loadings.T
        residual_sd = np.sqrt(np.maximum(1.0 - np.sum(loadings**2, axis=1), 0.12))
        roi_signals[idx] = (
            component[idx]
            + roi_ar[idx] * residual_sd[np.newaxis, :]
        )

    # Add a small shared low-frequency fluctuation and ROI-specific drift.
    time = np.arange(n_timepoints)
    slow_global = 0.12 * np.sin(2 * np.pi * time / 90.0)
    for roi in range(n_rois):
        roi_signals[:, roi] += slow_global + 0.04 * np.sin(
            2 * np.pi * time / (65.0 + 3.0 * roi) + roi * 0.25
        )

    # Scale to arbitrary BOLD-like signal units around a baseline.
    roi_signals = baseline + signal_scale * roi_signals

    roi_columns = [f"ROI_{i:02d}" for i in range(1, n_rois + 1)]
    signals_df = pd.DataFrame(roi_signals, columns=roi_columns)
    signals_df.insert(0, "time_seconds", time * tr_seconds)
    signals_df.insert(0, "timepoint", time + 1)

    metadata_df = pd.DataFrame([{
        "dataset_type": "synthetic",
        "n_timepoints": n_timepoints,
        "n_rois": n_rois,
        "TR_seconds": tr_seconds,
        "duration_seconds": n_timepoints * tr_seconds,
        "random_seed": random_seed,
        "baseline_arbitrary_units": baseline,
        "signal_scale": signal_scale,
        "state_1_timepoints": f"1-{b1}",
        "state_2_timepoints": f"{b1 + 1}-{b2}",
        "state_3_timepoints": f"{b2 + 1}-{n_timepoints}",
        "note": "Simulated data; not measured human fMRI and not for clinical use."
    }])

    ground_truth_df = pd.DataFrame({
        "timepoint": time + 1,
        "simulated_state": [f"state_{s + 1}" for s in states]
    })

    return signals_df, metadata_df, ground_truth_df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fMRI ROI time-series.")
    parser.add_argument("--timepoints", type=int, default=300)
    parser.add_argument("--rois", type=int, default=10)
    parser.add_argument("--tr", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/synthetic")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    signals, metadata, truth = generate_synthetic_fmri(
        n_timepoints=args.timepoints,
        n_rois=args.rois,
        tr_seconds=args.tr,
        random_seed=args.seed,
    )

    signals.to_csv(output_dir / "synthetic_fmri.csv", index=False)
    metadata.to_csv(output_dir / "synthetic_metadata.csv", index=False)
    truth.to_csv(output_dir / "synthetic_ground_truth.csv", index=False)

    print(f"Saved files to: {output_dir.resolve()}")
    print(f"Signal matrix shape (excluding time columns): {signals.shape[0]} x {args.rois}")
    print(signals.head())


if __name__ == "__main__":
    main()
