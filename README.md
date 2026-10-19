# AI Shot Evaluation

**Developer:** Evan Wright

## Abstract

Young lacrosse players model their shooting form after professionals, but have no
objective way to see how close they are. This project compares a video of a
player's shot to a professional's. A pretrained pose estimator turns each video
frame into body measurements. Machine learning models trained on those
measurements find the stages of the shot (load, step, release, follow-through).
Uniform-cost search then lines up the player's shot with the pro's, and the
matching key frames are compared to grade the shot and point out the biggest
differences in form.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# pretrained pose model (~30 MB)
mkdir -p data/models
curl -L -o data/models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

You will also need to make a `/data/raw` directory, and in it add the two reference videos. You can download from youtube here:

- [youtube.com/watch?v=Z1PEbo-Y8P4](https://www.youtube.com/watch?v=Z1PEbo-Y8P4)
- [youtube.com/watch?v=Isx\_91Uu4p4](http://youtube.com/watch?v=Isx_91Uu4p4) 

The shots used from each video are listed in `data/manifest.csv`.
