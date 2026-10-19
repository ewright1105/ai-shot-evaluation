import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

# MediaPipe landmark numbers used in this project (odd = left side, even = right side)
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
# every left landmark paired with its right twin (the nose has none)
LEFT_RIGHT_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)] + [(i, i + 1) for i in range(11, 33, 2)]


def fill_missing(landmarks):
    """
    Fills NaN frames from pose analyzation with evenly spaced values before/after gap
    for example: 0.42, NaN, NaN, 0.48 -> 0.42, 0.44, 0.46, 0.48
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


def normalize(world_lm):
    """
    put players at same size and location so shots can be compared from same point in world
    just set hip to 0,0,0 every time, flip y axis (mediapipe is inverted), and drop visibility (just confidence score)
    """
    pts = world_lm[..., :3].copy()
    pts[..., 1] *= -1

    hip_mid = (pts[:, L_HIP] + pts[:, R_HIP]) / 2
    shoulder_mid = (pts[:, L_SHOULDER] + pts[:, R_SHOULDER]) / 2
    # median over the whole clip, so one bad frame can't change the player's size
    torso = np.median(np.linalg.norm(shoulder_mid - hip_mid, axis=1))
    return (pts - hip_mid[:, None, :]) / torso


def mirror(pts):
    #flips lefty shooter to righty for comparison
    out = pts.copy()
    out[..., 0] *= -1
    for left, right in LEFT_RIGHT_PAIRS:
        out[:, [left, right]] = out[:, [right, left]]
    return out


def longest_gap(detected):
    longest = run = 0
    for found in detected:
        run = 0 if found else run + 1
        longest = max(longest, run)
    return longest


def main():
    for shot in csv.DictReader(open(MANIFEST)):
        raw = np.load(PROCESSED_DIR / f"{shot['shot_id']}_pose.npz")
        world_lm, detected = fill_missing(raw["world_lm"])
        pts = normalize(world_lm)
        side = shot["top_hand"]  # written in the manifest by watching the video
        if side == "left":
            pts = mirror(pts)  # Step 3.5: every shot is analyzed as if the top hand is the right hand

        out = PROCESSED_DIR / f"{shot['shot_id']}_clean.npz"
        np.savez(out, pts=pts, detected=detected, shooting_side=side, fps=raw["fps"])
        print(f"{shot['shot_id']:<15} filled {(~detected).sum():3d} of {len(detected)} frames "
              f"(longest gap {longest_gap(detected)}), top hand = {side:<5} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
