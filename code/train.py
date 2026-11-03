import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eda import load_shot
from label import STAGES
from preprocess import MANIFEST, ROOT

OUT_DIR = ROOT / "resources" / "results"

MODELS = {
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=0),
}


def load_dataset():
    # every frame of every shot stacked into one table
    X = []
    y = []
    groups = []
    for shot in csv.DictReader(open(MANIFEST)):
        features, stages = load_shot(shot["shot_id"])
        X.append(features)
        y.append(stages)
        groups += [shot["shot_id"]] * len(stages)
    return np.vstack(X), np.concatenate(y), np.array(groups)


def evaluate(model, X, y, groups):
    # train on 3 shots, test on the one left out, then repeat for each shot
    test_pred = np.empty_like(y)
    for shot in np.unique(groups):
        test = groups == shot
        train = ~test
        model.fit(X[train], y[train])
        test_pred[test] = model.predict(X[test])

        train_f1 = f1_score(y[train], model.predict(X[train]), average="macro")
        test_f1 = f1_score(y[test], test_pred[test], average="macro")
        print(f"  test on {shot}: train F1 = {train_f1:.2f}, test F1 = {test_f1:.2f}")
    return test_pred


def save_confusion_matrix(name, y, test_pred):
    ConfusionMatrixDisplay.from_predictions(y, test_pred, labels=STAGES, xticks_rotation=30, colorbar=False)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"confusion_{name}.png")
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, groups = load_dataset()
    print(len(X), "frames,", X.shape[1], "features")

    for name, model in MODELS.items():
        print("\n" + name)
        test_pred = evaluate(model, X, y, groups)
        print(f"  overall: test F1 = {f1_score(y, test_pred, average='macro'):.2f}, "f"accuracy = {accuracy_score(y, test_pred):.2f}")
        save_confusion_matrix(name, y, test_pred)


if __name__ == "__main__":
    main()
