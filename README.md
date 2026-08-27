# AR School Tour — room recognition

Markerless room recognition for an augmented-reality campus tour at Concordia
International School Hanoi.

A visitor wearing a Meta Quest 3 walks into a room. The headset reads its passthrough
camera, recognises which room it is with no QR code or marker on the wall, and anchors
an information panel to a real surface.

**This repository covers the machine-learning half only** — building the room dataset
and training the classifier that answers *which room is this?* The Unity application
that runs the model on the headset and draws the AR panels is separate work, outlined
in [`docs/charter.md`](docs/charter.md).

**Dataset:** [`viethaa/concordia-hanoi-rooms`](https://huggingface.co/datasets/viethaa/concordia-hanoi-rooms)

---

## Status

| | |
|---|---|
| Classes | `computer_science_room`, `math_room`, `physics_room`, `senior_lounge` |
| Frames | 134 (37 / 48 / 24 / 25) |
| Capture sessions | 1 |
| Best result | ~98% on a random frame split |
| Trustworthy? | **No — see below** |

That 98% is measured on frames randomly held out from the *same videos* used for
training. Consecutive frames of a continuous walk are near-identical, so the test set
contains near-copies of training images and the score is inflated. The tell is visible
in the training curves: train and held-out loss lie almost exactly on top of each
other, which independent data essentially never does.

The fix is a second capture session, not a model change. Once each room has been filmed
on two separate days, evaluation holds out an entire session and the number becomes
real. Reporting both figures — leaked and honest — is more informative than either alone.

---

## How it works

```
Quest 3 passthrough frame
        |
crop + resize to 224x224, rescale to [-1, 1]
        |
MobileNetV2 backbone (ImageNet weights, frozen)
        |
classification head   <- the only part trained on this school
        |
softmax -> per-room probabilities
        |
temporal vote over ~15 frames     <- suppresses flicker       (not yet built)
        |
confidence gate                   <- stays silent when unsure (not yet built)
        |
MRUK environment raycast -> wall position                     (not yet built)
        |
AR information panel                                          (not yet built)
```

The classifier answers *which room*. It says nothing about geometry, so scene
understanding decides where the panel goes.

---

## Layout

```
scripts/
  check_pretrained.py    baseline - stock ImageNet MobileNetV2 on any folder of images
  extract_frames.py      room videos -> labelled frames + metadata.csv
notebooks/
  01_baseline_mobilenet.ipynb   the baseline check, as a Colab notebook
  02_train_rooms.ipynb          transfer learning, metrics table, accuracy/loss curves
docs/
  charter.md             objectives, risks, 16-week schedule
```

Images are not stored in this repository. Frames live in the Hugging Face dataset; raw
video stays local under `videos_done/<session>/`.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Baseline

```bash
python scripts/check_pretrained.py              # ordinary photos
python scripts/check_pretrained.py ./photos/    # school photos
```

The stock model returns object names (`desk`, `monitor`) rather than room names. That
failure is the argument for transfer learning, and the baseline in the report.

### Build the dataset

Name each video after the room it shows — **the filename becomes the class label**:

```
videos/computer_science_room.mp4
videos/math_room.mp4
videos/physics_room.mp4
videos/senior_lounge.mp4
```

```bash
python scripts/extract_frames.py --videos ./videos --out ./data --session s1
```

Samples 2 frames/sec, discards motion-blurred frames, resizes to 512px, and writes
`data/<room>/<room>_s1_0001.jpg` plus `data/metadata.csv`.

For a repeat shoot of rooms already captured, keep the **same filenames** and increment
the session: `--session s2`. The session id is what separates the two shoots; a
different filename would create a new class.

`extract_frames.py` appends to `metadata.csv`. Move processed videos out of `videos/`
before re-running, or the same footage is recorded twice.

### Publish

```bash
hf upload viethaa/concordia-hanoi-rooms ./data . --repo-type dataset
```

### Train

Open `notebooks/02_train_rooms.ipynb` in Colab and run all. It reads the dataset from
the Hub, trains the head with the backbone frozen, fine-tunes the top 40 layers, and
outputs a metrics table (`metrics.csv`) plus accuracy and loss curves.

---

## Capture protocol

Two rules matter more for final accuracy than any model change.

**Film each room on separate occasions.** Frames from one continuous walk are
near-duplicates. Each shoot gets its own `--session` id; evaluation holds out a whole
session.

**Cover the room, not one viewpoint.** 60-90 seconds per room: walk the perimeter,
pause at three or four standing positions, face each wall, and include the doorway view
— that is the first thing the headset sees. A 20-second clip yields ~37 frames from
essentially one position.

Still needed: an `other` class of corridors, stairwells, doorways and outdoor space.
Without it, a softmax over N rooms cannot express uncertainty, and every hallway is
confidently assigned to whichever room it least resembles.

---

## Dataset schema

ImageFolder layout plus `metadata.csv` with `file_name`, `label`, `session`,
`source_video`, `timestamp_s`, `sharpness`.

Because `metadata.csv` is present, `datasets` returns `label` as a **string**, not an
integer `ClassLabel`. The training notebook builds the class index itself.

---

## Note on the data

The dataset contains interior imagery of a school. Rooms are filmed empty and no
identifiable person appears in any frame. Publication is subject to school approval.
