# Phase 1: Synthetic fMRI dataset

Files:
- synthetic_fmri.csv: 300 time points, 10 ROI BOLD-like signals, plus timepoint/time_seconds.
- synthetic_metadata.csv: generation settings and important caveats.
- synthetic_ground_truth.csv: known simulated state labels; use only for validating methods.

Run from the project root:
    python src/data_loader/synthetic_generator.py

Optional:
    python src/data_loader/synthetic_generator.py --timepoints 300 --rois 10 --tr 2.0 --seed 42

Install:
    pip install numpy pandas

The values are synthetic arbitrary BOLD-like units, not measured fMRI and not clinical data.
