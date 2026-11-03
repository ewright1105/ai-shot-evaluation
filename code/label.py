import csv
import sys

import cv2

from preprocess import PROCESSED_DIR, ROOT

LABELS_DIR = ROOT / "data" / "labels"
STAGES = ["Other", "Load", "Step", "Release", "Follow-through"]

def stage_per_frame(marks, num_frames):
    stages = []
    current = "Other"
    for i in range(num_frames):
        if i in marks:
            current = marks[i]
        stages.append(current)
    return stages


def label_shot(shot_id):
    cap = cv2.VideoCapture(str(PROCESSED_DIR / f"{shot_id}_check.mp4"))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    marks = {}
    i = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        stage = stage_per_frame(marks, i + 1)[i]
        cv2.putText(frame, stage, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
        cv2.imshow(shot_id, cv2.resize(frame, (1280, 720)))

        key = chr(cv2.waitKey(0) & 0xFF)
        if key == "a":
            i -= 1
        elif key == "d":
            i += 1
        elif key == "q":
            i -= 10
        elif key == "e":
            i += 10
        elif key in "01234":
            marks[i] = STAGES[int(key)]
        elif key == "s":
            save(shot_id, stage_per_frame(marks, num_frames))
            break
        elif key == "\x1b":  # esc
            break
        i = max(0, min(i, num_frames - 1))  # stay inside the video

    cap.release()
    cv2.destroyAllWindows()


def save(shot_id, stages):
    LABELS_DIR.mkdir(exist_ok=True)
    with open(LABELS_DIR / f"{shot_id}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "stage"])
        for i, stage in enumerate(stages):
            writer.writerow([i, stage])
    print("saved labels for", shot_id)


if __name__ == "__main__":
    label_shot(sys.argv[1])
