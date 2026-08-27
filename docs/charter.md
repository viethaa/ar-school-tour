# Project charter — AR School Tour

Autumn term 2026. Concordia International School Hanoi.

A visitor wearing a Quest 3 walks into a room. The headset sees it through the
passthrough camera, recognises it without any QR code on the wall, and puts an
information panel where the panel belongs.

| | |
|---|---|
| Device | Meta Quest 3 (Horizon OS v74+) |
| Model | MobileNetV2, transfer learning |
| Runtime | Unity 6 + Unity Inference Engine (formerly Sentis) |
| Term | Aug-Dec 2026 |

---

## Methodology

| Stage | What happens | The question it answers |
|---|---|---|
| **Baseline** | Stock ImageNet MobileNetV2 on ordinary photos, then on school rooms | Does the toolchain work — and can an off-the-shelf model already do this? |
| **Customise** | Freeze the backbone, train a new head, fine-tune the top layers | Can a small amount of local data teach a general vision model a specific building? |
| **Evaluate** | Test on a capture session never seen before. Confusion matrix, threshold sweep, on-device latency | How wrong is it, in what direction, and is that tolerable to a visitor? |
| **Deploy** | Export to ONNX -> `.sentis`, quantise, run in Unity on the headset | Does any of this survive contact with a real device and a real person? |

**One change to that order.** Deploy is written last but must be *attempted* first.
Prove that some model — even a deliberately terrible two-class one — runs on the Quest
early. The Keras -> ONNX -> `.sentis` conversion is where this class of project usually
dies, and finding that out in December instead of September is the most expensive
mistake available.

---

## Success criteria

The headline criterion is the last row, not the first.

| Criterion | Target | Measured how |
|---|---|---|
| Room accuracy | >= 90% | Top-1 on a held-out capture session, not a random frame split |
| False panels | <= 10% | Corridor / stairwell / outdoor frames that wrongly trigger a room |
| Inference latency | <= 100 ms | Per frame, on the headset, not on a laptop |
| Model size | <= 5 MB | Quantised `.sentis` file |
| Time to label | <= 2 s | Entering the room to panel appearing |
| Label stability | 0 flips | Over a 30-second dwell in one room |
| **Unaided completion** | **5 of 5** | **First-time visitors finish the tour without a human guide** |

---

## Known risks

**A softmax over N rooms can never say "I don't know."** The probabilities always sum
to 1, so a stairwell is reported as a room with high confidence. On a walking tour most
frames are not of any target room, making this the failure a visitor meets first.
*Fix:* an explicit `other` class with varied negatives, plus a confidence gate.

**The device is last in the plan and first to fail.** Training in Colab is well-trodden;
getting a Keras model through ONNX, into Inference Engine format, quantised, and running
asynchronously on a mobile chip is not. *Fix:* vertical slice early.

**A random train/test split reports a fake number.** Frames from one continuous video
are near-identical, so a random split leaks training images into the test set.
*Fix:* capture on >= 3 separate occasions; hold out an entire session.

**Phone photos are not Quest passthrough frames.** Different resolution, colour,
exposure, distortion and motion blur. *Fix:* capture through the Quest where possible;
augment aggressively; keep a headset-captured validation set.

**Classification gives the room, not the geometry.** A naive implementation welds the
panel to the centre of the view like a HUD. *Fix:* the classifier chooses *what*, MRUK
scene understanding chooses *where*.

**Quest 3 may be the wrong delivery device for an actual tour.** ~2 hours of battery,
one visitor at a time, and a person walking a school with mediated vision raises
supervision questions. *Fix:* decide deliberately between a supervised flagship demo
and a phone build for visitors at scale.

---

## Schedule

| Phase | Dates | Work | Ships |
|---|---|---|---|
| 0 Ground | Aug 31 - Sep 11 | Toolchain, school permission, device check | Working toolchain, signed permission |
| 1 Vertical slice | Sep 14 - Oct 2 | Two rooms, throwaway model, Keras -> ONNX -> `.sentis`, on headset | A label on the headset display |
| 2 Dataset | Oct 5 - Oct 23 | Capture protocol, 3+ sessions per room, `other` class | Versioned dataset + datasheet |
| 3 Model | Oct 26 - Nov 13 | Transfer learning, fine-tuning, held-out evaluation | Trained model + results section |
| 4 Experience | Nov 16 - Dec 4 | Temporal voting, confidence gating, MRUK anchoring, room content | End-to-end tour |
| 5 Pilot | Dec 7 - Dec 18 | Five first-time visitors walk the tour | Pilot results + final report |

If a phase slips, the next starts anyway with whatever exists. A working tour of three
rooms beats an unfinished tour of eight.

---

## What is still unbuilt

The classifier answers *which room*. Everything below is Unity/C# work and is the
larger half of the project by hours:

- Unity project + Meta XR SDK setup
- Passthrough Camera API frame capture (Meta ships a sample)
- Keras -> ONNX -> `.sentis` conversion, quantised
- Per-frame inference via Unity Inference Engine (Meta ships a sample)
- Temporal vote over ~15 frames
- Confidence gate
- MRUK raycast to place the panel on a real surface (Meta ships a sample)
- Panel UI, room content, tour flow
- Build and deploy to headset

---

## References

- [Passthrough Camera API in Unity](https://developers.meta.com/horizon/documentation/unity/unity-pca-documentation/)
- [Unity Inference Engine for on-device ML/CV models](https://developers.meta.com/horizon/documentation/unity/unity-pca-sentis/)
- [Unity Passthrough Camera API samples](https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples)
- [TensorFlow transfer learning tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning)
