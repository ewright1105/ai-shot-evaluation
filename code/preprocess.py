"""Phase 3: clean up the raw pose from pose.py.

Reads data/processed/<shot_id>_pose.npz and saves data/processed/<shot_id>_clean.npz:

    world_lm  (T, 33, 4)  world landmarks with gaps filled in
    detected  (T,)        True where MediaPipe really found the person
    fps

Run from the repo root (after pose.py):
    .venv/bin/python code/preprocess.py
"""

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.csv"
PROCESSED_DIR = ROOT / "data" / "processed"


def fill_missing(landmarks):
    """Step 3.2: fill frames where no person was found by linear interpolation.

    Each number (joint x, y, z, visibility) is treated as its own time series,
    and missing frames get the value on the straight line between the nearest
    detected frames before and after. Gaps at the very start or end are filled
    with the nearest detected value.

    Returns the filled array and a mask of which frames were real detections.
    """
    detected = ~np.isnan(landmarks[:, 0, 0])
    if not detected.any():
        raise ValueError("no person detected in any frame")

    filled = landmarks.copy()
    frames = np.arange(len(landmarks))
    flat = filled.reshape(len(filled), -1)  # (T, 33*4): one column per time series
    for col in range(flat.shape[1]):
        flat[:, col] = np.interp(frames, frames[detected], flat[detected, col])
    return filled, detected


def longest_gap(detected):
    """Length of the longest run of missing frames."""
    longest = run = 0
    for found in detected:
        run = 0 if found else run + 1
        longest = max(longest, run)
    return longest


def main():
    for shot in csv.DictReader(open(MANIFEST)):
        raw = np.load(PROCESSED_DIR / f"{shot['shot_id']}_pose.npz")
        world_lm, detected = fill_missing(raw["world_lm"])

        out = PROCESSED_DIR / f"{shot['shot_id']}_clean.npz"
        np.savez(out, world_lm=world_lm, detected=detected, fps=raw["fps"])
        print(f"{shot['shot_id']:<15} filled {(~detected).sum():3d} of {len(detected)} frames "
              f"(longest gap {longest_gap(detected)}) -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
