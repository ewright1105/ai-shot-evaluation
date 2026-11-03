import csv

import matplotlib
matplotlib.use("Agg")  # save images, don't open windows
import matplotlib.pyplot as plt
import numpy as np

from features import FEATURE_NAMES
from label import LABELS_DIR, STAGES
from preprocess import MANIFEST, PROCESSED_DIR, ROOT

OUT_DIR = ROOT / "resources" / "eda"
COLORS = {"Other": "lightgray", "Load": "tab:blue", "Step": "tab:orange",
          "Release": "tab:red", "Follow-through": "tab:green"}
# a few features worth watching over time, one per row of the timeline plot
TIMELINE_FEATURES = ["top_hand_height", "stride", "top_hand_speed", "front_foot_speed"]


def load_shot(shot_id):
    features = np.load(PROCESSED_DIR / f"{shot_id}_features.npz")["features"]
    stages = [row["stage"] for row in csv.DictReader(open(LABELS_DIR / f"{shot_id}.csv"))]
    return features, np.array(stages)


def print_stage_counts(shots):
    print(f"{'shot':<16}" + "".join(f"{s:>16}" for s in STAGES) + f"{'total':>8}")
    totals = {s: 0 for s in STAGES}
    for shot_id, (_, stages) in shots.items():
        counts = {s: int((stages == s).sum()) for s in STAGES}
        for s in STAGES:
            totals[s] += counts[s]
        print(f"{shot_id:<16}" + "".join(f"{counts[s]:>16}" for s in STAGES) + f"{len(stages):>8}")
    all_frames = sum(totals.values())
    print(f"{'all':<16}" + "".join(f"{totals[s]:>11} ({100 * totals[s] / all_frames:2.0f}%)" for s in STAGES)
          + f"{all_frames:>8}")


def plot_timeline(shot_id, features, stages):
    # each feature over the shot, background shaded by the labeled stage
    fig, axes = plt.subplots(len(TIMELINE_FEATURES), 1, figsize=(10, 8), sharex=True)
    for ax, name in zip(axes, TIMELINE_FEATURES):
        ax.plot(features[:, FEATURE_NAMES.index(name)], color="black", linewidth=1)
        for i, stage in enumerate(stages):
            ax.axvspan(i - 0.5, i + 0.5, color=COLORS[stage], alpha=0.3, linewidth=0)
        ax.set_ylabel(name, fontsize=8)
    axes[-1].set_xlabel("frame")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s], alpha=0.3) for s in STAGES]
    fig.legend(handles, STAGES, loc="upper center", ncol=len(STAGES))
    fig.suptitle(shot_id, y=0.94)
    fig.savefig(OUT_DIR / f"timeline_{shot_id}.png", dpi=120)
    plt.close(fig)


def plot_feature_by_stage(shots):
    # one box per stage for every feature, all shots pooled
    features = np.vstack([f for f, _ in shots.values()])
    stages = np.concatenate([s for _, s in shots.values()])
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, name in zip(axes.flat, FEATURE_NAMES):
        col = features[:, FEATURE_NAMES.index(name)]
        ax.boxplot([col[stages == s] for s in STAGES], showfliers=False)
        ax.set_xticks(range(1, len(STAGES) + 1), [s[:7] for s in STAGES], fontsize=7)
        ax.set_title(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "features_by_stage.png", dpi=120)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shots = {s["shot_id"]: load_shot(s["shot_id"]) for s in csv.DictReader(open(MANIFEST))}
    print_stage_counts(shots)
    for shot_id, (features, stages) in shots.items():
        plot_timeline(shot_id, features, stages)
    plot_feature_by_stage(shots)
    print(f"\nplots -> {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
