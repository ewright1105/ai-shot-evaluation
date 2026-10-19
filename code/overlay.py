import csv

import cv2
import numpy as np

from preprocess import MANIFEST, PROCESSED_DIR, ROOT

RAW_DIR = ROOT / "data" / "raw"
# pairs of landmarks to connect with a line (via mediapipe docs)
SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # shoulders and arms
    (11, 23), (12, 24), (23, 24),                      # torso
    (23, 25), (25, 27), (24, 26), (26, 28),            # legs
    (27, 31), (28, 32),                                # feet
]
GREEN, RED, YELLOW = (0, 255, 0), (0, 0, 255), (0, 255, 255)  # cv2 works in BGR not RGB


def write_overlay(video_path, start_s, image_lm, out_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(start_s * fps))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    for i, frame_lm in enumerate(image_lm):
        ok, frame = cap.read()
        if not ok:
            break
        if np.isnan(frame_lm[0, 0]):
            cv2.putText(frame, "FILLED", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
        else:
            xy = frame_lm[:, :2].astype(int)
            for a, b in SKELETON:
                cv2.line(frame, tuple(xy[a]), tuple(xy[b]), GREEN, 2)
            for x, y in xy[11:29]:
                cv2.circle(frame, (x, y), 4, RED, -1)
        cv2.putText(frame, f"frame {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, YELLOW, 2)
        writer.write(frame)

    cap.release()
    writer.release()


def main():
    for shot in csv.DictReader(open(MANIFEST)):
        pose = np.load(PROCESSED_DIR / f"{shot['shot_id']}_pose.npz")
        out = PROCESSED_DIR / f"{shot['shot_id']}_check.mp4"
        write_overlay(RAW_DIR / f"{shot['video_id']}.mp4", float(shot["start_s"]), pose["image_lm"], out)
        print(f"{shot['shot_id']:<15} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
