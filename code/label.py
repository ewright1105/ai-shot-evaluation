import csv
import sys

import cv2

from preprocess import PROCESSED_DIR, ROOT

LABELS_DIR = ROOT / "data" / "labels"
STAGES = ["Other", "Load", "Step", "Release", "Follow-through"]
# keys: a/d = back/forward 1 frame, q/e = back/forward 10 frames
#       1-4 = this frame starts Load/Step/Release/Follow-through, 0 = shot is over (back to Other)
#       s = save and quit, esc = quit without saving
MOVES = {ord("a"): -1, ord("d"): 1, ord("q"): -10, ord("e"): 10}


def stage_per_frame(marks, num_frames):
    # marks = {frame: stage}; each mark holds until the next one
    stages, current = [], "Other"
    for i in range(num_frames):
        current = marks.get(i, current)
        stages.append(current)
    return stages


def label_shot(shot_id):
    cap = cv2.VideoCapture(str(PROCESSED_DIR / f"{shot_id}_check.mp4"))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    marks, i = {}, 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        stage = stage_per_frame(marks, i + 1)[-1]
        cv2.putText(frame, stage, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
        cv2.imshow(shot_id, cv2.resize(frame, None, fx=720 / frame.shape[0], fy=720 / frame.shape[0]))

        key = cv2.waitKey(0) & 0xFF
        if key in MOVES:
            i = min(max(i + MOVES[key], 0), num_frames - 1)
        elif ord("0") <= key <= ord("4"):
            marks[i] = STAGES[key - ord("0")]
        elif key == ord("s"):
            save(shot_id, stage_per_frame(marks, num_frames))
            break
        elif key == 27:  # esc
            break

    cap.release()
    cv2.destroyAllWindows()


def save(shot_id, stages):
    LABELS_DIR.mkdir(exist_ok=True)
    out = LABELS_DIR / f"{shot_id}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "stage"])
        writer.writerows(enumerate(stages))
    counts = {s: stages.count(s) for s in STAGES}
    print(f"{shot_id} -> {out.relative_to(ROOT)}  {counts}")


if __name__ == "__main__":
    label_shot(sys.argv[1])
