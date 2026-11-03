import csv

import matplotlib.pyplot as plt
import numpy as np

from features import FEATURE_NAMES
from label import LABELS_DIR, STAGES
from preprocess import MANIFEST, PROCESSED_DIR, ROOT

OUT_DIR = ROOT / "resources" / "eda"
COLORS = {"Other": "gray", "Load": "blue", "Step": "orange", "Release": "red", "Follow-through": "green"}
TIMELINE_FEATURES = ["top_hand_height", "stride", "top_hand_speed", "front_foot_speed"]


def load_shot(shot_id):
    features = np.load(PROCESSED_DIR / f"{shot_id}_features.npz")["features"]
    stages = []
    for row in csv.DictReader(open(LABELS_DIR / f"{shot_id}.csv")):
        stages.append(row["stage"])
    return features, np.array(stages)


def print_stage_counts(all_stages):
    print("frames per stage:")
    for stage in STAGES:
        count = np.sum(all_stages == stage)
        print(f"  {stage}: {count} ({100 * count / len(all_stages):.0f}%)")


def plot_timeline(shot_id, features, stages):
    frames = np.arange(len(stages))
    fig, axes = plt.subplots(len(TIMELINE_FEATURES), 1, figsize=(10, 8), sharex=True)
    for ax, name in zip(axes, TIMELINE_FEATURES):
        values = features[:, FEATURE_NAMES.index(name)]
        ax.plot(frames, values, color="black", linewidth=0.5)
        for stage in STAGES:
            ax.scatter(frames[stages == stage], values[stages == stage], color=COLORS[stage], s=5, label=stage)
        ax.set_ylabel(name)
    axes[0].legend()
    axes[0].set_title(shot_id)
    axes[-1].set_xlabel("frame")
    plt.savefig(OUT_DIR / f"timeline_{shot_id}.png")
    plt.close()


def plot_feature_by_stage(features, stages):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, name in zip(axes.flat, FEATURE_NAMES):
        values = features[:, FEATURE_NAMES.index(name)]
        ax.boxplot([values[stages == stage] for stage in STAGES], showfliers=False)
        ax.set_xticklabels(STAGES, fontsize=7)
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "features_by_stage.png")
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_features = []
    all_stages = []
    for shot in csv.DictReader(open(MANIFEST)):
        features, stages = load_shot(shot["shot_id"])
        plot_timeline(shot["shot_id"], features, stages)
        all_features.append(features)
        all_stages.append(stages)

    all_features = np.vstack(all_features)
    all_stages = np.concatenate(all_stages)
    print_stage_counts(all_stages)
    plot_feature_by_stage(all_features, all_stages)
    print("plots saved to resources/eda")


if __name__ == "__main__":
    main()
