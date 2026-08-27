<div align="center">

# AR School Tour

**An augmented reality tour of the Concordia Hanoi campus, built for the Meta Quest 3.**
**The headset recognises each room you walk into and shows information about it.**

</div>

---

## Overview

Most AR tours need a marker on the wall. You scan a QR code, and the app tells you where
you are. This one doesn't.

Instead, the headset looks at the room through its own camera and works out where it is
from what it sees: the shape of the space, the furniture, the equipment. Walk into the
physics room and a panel appears explaining what happens there. Walk into the senior
lounge and it changes. Nothing is stuck to the walls.

Teaching a computer to tell one classroom from another is the hard part, and that is
what this repository does.

---

## Method

**1. Film each room**

A slow walk through the room with a camera, covering it from several positions.

**2. Turn the video into training images**

Video is just a lot of still pictures. We pull out a couple of frames per second, throw
away the blurry ones, and sort them into a folder per room. Those folders become the
labels: everything in the `physics_room` folder is, by definition, the physics room.

**3. Teach a model to recognise the rooms**

We start from **MobileNetV2**, a model that has already learned to see. It was trained on
over a million photographs, so it understands edges, textures, furniture and layout, but
it has no idea what a *physics room* is.

Rather than teach it to see from scratch, we keep everything it already knows and train
one small new layer on top, using our photos. This is called **transfer learning**, and
it is why a few hundred pictures are enough where you would otherwise need millions.

**4. Run it on the headset**

```
camera sees the room  →  model names the room  →  panel appears
```

MobileNetV2 was designed to be small and fast enough to run on a phone, which is exactly
what a headset needs. The model runs on the Quest itself, so the tour works without a
server or an internet connection.

---

## What's in this repo

| | |
|---|---|
| `scripts/extract_frames.py` | Turns room videos into labelled training images |
| `notebooks/train_rooms.ipynb` | Trains the model and measures how well it does |
| `docs/charter.md` | Goals, risks and schedule for the wider project |

The photos themselves are not stored here. They live in the dataset:
**[viethaa/concordia-hanoi-rooms](https://huggingface.co/datasets/viethaa/concordia-hanoi-rooms)**

---

## Status

The model currently tells four rooms apart: computer science, maths, physics, and the
senior lounge. The AR side, showing the panels on the headset, is still being built.

---

<div align="center">
<sub>Concordia International School Hanoi</sub>
</div>
