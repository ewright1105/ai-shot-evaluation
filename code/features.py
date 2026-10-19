import csv

import numpy as np

from preprocess import (
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, MANIFEST, NOSE, PROCESSED_DIR,
    R_ANKLE, R_ELBOW, R_HIP, R_SHOULDER, R_WRIST, ROOT,
)

# some random points to look at when comparing
FEATURE_NAMES = [
    "top_elbow_height",   # top elbow height - top shoulder height
    "top_elbow_angle",    # shoulder-elbow-wrist angle
    "top_hand_height",    # top wrist height - nose height
    "hands_from_body",    # distance from hand midpoint to shoulder midpoint
    "stride",             # distance between ankles
    "front_knee_angle",   # hip-knee-ankle angle of front leg
    "hip_shoulder_sep",   # twist of the shoulder line vs the hip line, seen from above
    "shoulder_tilt",      # shoulder line vs horizontal (+ = top shoulder higher)
    "body_lean",          # torso angle from vertical
    "hand_separation",    # distance between wrists
    "top_hand_speed",     # top wrist movement per frame / fastest frame
    "front_foot_speed",   # front ankle movement per frame / fastest frame
]


def joint_angle(a, b, c):
    # a, b, c can be hand/elbow/shoulder or ankle/knee/hip
    ba, bc = a - b, c - b
    cos = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def heading(a, b):
    # direction fron a -> b
    d = b - a
    return np.degrees(np.arctan2(d[:, 2], d[:, 0]))


def relative_speed(points):
    speed = np.linalg.norm(np.gradient(points, axis=0), axis=1)
    return speed / speed.max()


def compute_features(pts):
    p = lambda i: pts[:, i]  # noqa: E731  (shorthand: all frames of landmark i)
    shoulder_mid = (p(L_SHOULDER) + p(R_SHOULDER)) / 2  # the hips are already at (0, 0, 0)
    hands_mid = (p(L_WRIST) + p(R_WRIST)) / 2
    shoulder_line = p(R_SHOULDER) - p(L_SHOULDER)

    twist = heading(p(L_SHOULDER), p(R_SHOULDER)) - heading(p(L_HIP), p(R_HIP))
    columns = {
        "top_elbow_height": p(R_ELBOW)[:, 1] - p(R_SHOULDER)[:, 1],
        "top_elbow_angle": joint_angle(p(R_SHOULDER), p(R_ELBOW), p(R_WRIST)),
        "top_hand_height": p(R_WRIST)[:, 1] - p(NOSE)[:, 1],
        "hands_from_body": np.linalg.norm(hands_mid - shoulder_mid, axis=1),
        "stride": np.linalg.norm(p(L_ANKLE) - p(R_ANKLE), axis=1),
        "front_knee_angle": joint_angle(p(L_HIP), p(L_KNEE), p(L_ANKLE)),
        "hip_shoulder_sep": (twist + 180) % 360 - 180,  # keep it between -180 and 180
        "shoulder_tilt": np.degrees(np.arcsin(shoulder_line[:, 1] / np.linalg.norm(shoulder_line, axis=1))),
        "body_lean": np.degrees(np.arccos(shoulder_mid[:, 1] / np.linalg.norm(shoulder_mid, axis=1))),
        "hand_separation": np.linalg.norm(p(L_WRIST) - p(R_WRIST), axis=1),
        "top_hand_speed": relative_speed(p(R_WRIST)),
        "front_foot_speed": relative_speed(p(L_ANKLE)),
    }
    return np.column_stack([columns[name] for name in FEATURE_NAMES])


def main():
    for shot in csv.DictReader(open(MANIFEST)):
        clean = np.load(PROCESSED_DIR / f"{shot['shot_id']}_clean.npz")
        features = compute_features(clean["pts"])
        out = PROCESSED_DIR / f"{shot['shot_id']}_features.npz"
        np.savez(out, features=features, feature_names=np.array(FEATURE_NAMES),
                 detected=clean["detected"], fps=clean["fps"])

        print(f"\n{shot['shot_id']} ({len(features)} frames) -> {out.relative_to(ROOT)}")
        print(f"  {'feature':<18}{'min':>8}{'max':>8}{'mean':>8}")
        for name, col in zip(FEATURE_NAMES, features.T):
            print(f"  {name:<18}{col.min():>8.2f}{col.max():>8.2f}{col.mean():>8.2f}")


if __name__ == "__main__":
    main()
