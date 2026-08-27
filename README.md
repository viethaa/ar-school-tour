# AR School Tour — Room Recognition

Markerless room recognition for an augmented-reality campus tour at Concordia
International School Hanoi. This repository turns video walkthroughs of school rooms
into a labelled image dataset, then fine-tunes MobileNetV2 to identify which room a
Meta Quest 3 is looking at from its passthrough camera alone — no QR codes, no markers
on the wall.

| | |
|---|---|
| **Task** | 4-class indoor scene recognition |
| **Model** | MobileNetV2 (ImageNet backbone, frozen) + trained head |
| **Dataset** | [viethaa/concordia-hanoi-rooms](https://huggingface.co/datasets/viethaa/concordia-hanoi-rooms) · 134 frames · 1 session |
| **Target runtime** | Unity 6 + Inference Engine, on Quest 3 |

> **Scope.** This repo is the machine-learning half: dataset and classifier. The Unity
> application that runs the model on the headset and draws the AR panels is separate
> work — see [`docs/charter.md`](docs/charter.md).

---

## Quick start

```bash
# 1. install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. confirm the toolchain, and that a stock model can't do this
python scripts/check_pretrained.py

# 3. build the dataset  (video filename = class label)
python scripts/extract_frames.py --videos ./videos --out ./data --session s1

# 4. publish
hf upload viethaa/concordia-hanoi-rooms ./data . --repo-type dataset
```

Then open `notebooks/train_rooms.ipynb` in Colab and run all.

---

## What's in here

| File | Purpose |
|---|---|
| `scripts/check_pretrained.py` | Runs stock ImageNet MobileNetV2 on any folder. Establishes the baseline: it returns `desk` and `monitor`, never room names. |
| `scripts/extract_frames.py` | Video → labelled frames. Samples 2 fps, drops blurred frames, writes `metadata.csv` with the capture session. |
| `notebooks/train_rooms.ipynb` | Transfer learning + evaluation. Outputs `metrics.csv`, accuracy and loss curves. |
| `docs/charter.md` | Objectives, risks, 16-week schedule, and what remains unbuilt. |
| `requirements.txt` | Pinned minimums for the scripts. |

Images are never committed here. Frames live on Hugging Face; raw video stays local.

---

## Pipeline

```
passthrough frame → 224×224 → MobileNetV2 (frozen) → trained head → softmax
                                                                      ↓
                                       temporal vote → confidence gate → AR panel
                                       └────────── not yet built ──────────┘
```

The classifier answers *which room*. It knows nothing about geometry, so scene
understanding decides where the panel goes.

---

## Results

| | |
|---|---|
| Accuracy | ~98% |
| Split | random frames, single session |
| **Trustworthy** | **No** |

The test frames come from the same videos as the training frames. Consecutive frames of
a walkthrough are near-identical, so the test set holds near-copies of training images.
The giveaway is in the loss curves: train and held-out lie almost exactly on top of each
other, which independent data never does.

This is fixed by capturing a second session, not by changing the model. With two
sessions, evaluation holds out an entire day and the number becomes real.

---

## Capture protocol

Two rules affect final accuracy more than any model change.

1. **Film each room on separate days.** Each shoot gets its own `--session` id, keeping
   the same video filenames. Evaluation then holds out a whole session.
2. **Cover the room, not one viewpoint.** 60–90 seconds: walk the perimeter, pause at
   three or four positions, face each wall, include the doorway view. A 20-second clip
   gives ~37 frames from essentially one spot.

Still missing: an `other` class of corridors, stairwells and outdoor space. Without it a
softmax over four rooms cannot express uncertainty, and every hallway gets confidently
labelled as a room.

---

## Roadmap

| Phase | Status |
|---|---|
| Dataset pipeline | done |
| Baseline + transfer learning | done |
| Second capture session, `other` class | next |
| Export to `.sentis`, run on Quest 3 | |
| Temporal voting, confidence gate, MRUK anchoring | |
| Pilot with first-time visitors | |

---

## Data note

The dataset contains interior imagery of a school. Rooms are filmed empty and no
identifiable person appears in any frame. Publication is subject to school approval.
