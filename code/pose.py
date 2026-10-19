import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.csv"
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
MODEL = ROOT / "data" / "models" / "pose_landmarker_heavy.task"
NUM_LANDMARKS = 33


def extract_pose(video_path, start_s, end_s):=
    # open video and read properties
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # get num of frames 
    first, last = round(start_s * fps), round(end_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    # initialize model to find people in a video
    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    ))

    image_lm, world_lm = [], [] # image landmark is where joint is on screen (pixels and world landmark is where joint is relative to hips
    for idx in range(first, last):
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # MediaPipe wants RGB
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(idx * 1000 / fps))
        image_frame = np.full((NUM_LANDMARKS, 4), np.nan)
        world_frame = np.full((NUM_LANDMARKS, 4), np.nan)
        if result.pose_landmarks:
            for j, lm in enumerate(result.pose_landmarks[0]):
                image_frame[j] = (lm.x * width, lm.y * height, lm.z * width, lm.visibility)
            for j, lm in enumerate(result.pose_world_landmarks[0]):
                world_frame[j] = (lm.x, lm.y, lm.z, lm.visibility)
        image_lm.append(image_frame)
        world_lm.append(world_frame)

    cap.release()
    landmarker.close()
    return np.stack(image_lm), np.stack(world_lm), fps


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for shot in csv.DictReader(open(MANIFEST)):
        video = RAW_DIR / f"{shot['video_id']}.mp4"
        image_lm, world_lm, fps = extract_pose(video, float(shot["start_s"]), float(shot["end_s"]))
        out = OUT_DIR / f"{shot['shot_id']}_pose.npz"
        np.savez(out, image_lm=image_lm, world_lm=world_lm, fps=fps)

        found = ~np.isnan(world_lm[:, 0, 0])
        print(f"{shot['shot_id']:<15} {len(world_lm):4d} frames, person found in {found.sum():4d} "
              f"({100 * found.mean():.0f}%) -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
