"""
AR School Tour — Step 3: turn room videos into a labelled image dataset.

WHAT THIS DOES
  Reads a folder of videos named after rooms, samples frames at a fixed rate,
  drops the blurry ones, resizes them, and writes an ImageFolder-style dataset
  plus a metadata.csv that records WHICH CAPTURE SESSION each frame came from.

  That session column is the whole point. Frames from one continuous video are
  near-identical, so a random train/test split would put near-copies of your
  training images into your test set and hand you a meaningless 99%. Recording
  the session lets you hold out an entire session later, which is the only
  split that tells you the truth.

INPUT   videos/library.mov, videos/science_lab.mp4, videos/gym.mov, ...
        (filename = class label; spaces and hyphens become underscores)

OUTPUT  data/library/library_s1_0001.jpg
        data/science_lab/science_lab_s1_0001.jpg
        data/metadata.csv

USAGE
  pip install opencv-python
  python 02_extract_frames.py --videos ./videos --out ./data --session s1

  # a second capture on a different day:
  python 02_extract_frames.py --videos ./videos_day2 --out ./data --session s2

ARGUMENTS WORTH TUNING
  --fps        frames sampled per second of video (default 2)
  --blur-drop  discard frames blurrier than this; raise to be stricter (default 40)
  --max-side   longest edge in pixels (default 512 — leaves room to augment later)
"""

import argparse
import csv
import os
import sys

try:
    import cv2
except ImportError:
    sys.exit("OpenCV missing.  Run:  pip install opencv-python")

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mpg", ".mpeg", ".webm"}


def label_from_filename(path):
    """videos/Science Lab.MOV -> science_lab"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.strip().lower().replace(" ", "_").replace("-", "_")


def sharpness(gray):
    """Variance of the Laplacian. Low = blurry. Standard cheap blur metric."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def resize_max_side(img, max_side):
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return cv2.resize(
        img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
    )


def extract_one(video_path, out_root, session, target_fps, blur_drop, max_side):
    label = label_from_filename(video_path)
    out_dir = os.path.join(out_root, label)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  !! could not open {video_path} — see the HEVC note at the bottom")
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Sample every Nth frame rather than decoding-and-discarding by timestamp.
    step = max(1, int(round(src_fps / target_fps)))

    rows, kept, blurry, idx = [], 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            s = sharpness(gray)
            if s < blur_drop:
                blurry += 1
            else:
                kept += 1
                frame = resize_max_side(frame, max_side)
                name = f"{label}_{session}_{kept:04d}.jpg"
                cv2.imwrite(
                    os.path.join(out_dir, name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]
                )
                rows.append(
                    {
                        "file_name": f"{label}/{name}",
                        "label": label,
                        "session": session,
                        "source_video": os.path.basename(video_path),
                        "timestamp_s": round(idx / src_fps, 2),
                        "sharpness": round(s, 1),
                    }
                )
        idx += 1
    cap.release()

    dur = total / src_fps if total else 0
    print(
        f"  {label:<16} {dur:5.1f}s @ {src_fps:.0f}fps  ->  "
        f"{kept:4d} kept, {blurry:3d} dropped as blurry"
    )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="./videos", help="folder of room videos")
    ap.add_argument("--out", default="./data", help="dataset output folder")
    ap.add_argument("--session", default="s1", help="capture session id, e.g. s1")
    ap.add_argument("--fps", type=float, default=2.0, help="frames sampled per second")
    ap.add_argument("--blur-drop", type=float, default=40.0, help="sharpness floor")
    ap.add_argument("--max-side", type=int, default=512, help="longest edge, pixels")
    args = ap.parse_args()

    if not os.path.isdir(args.videos):
        sys.exit(f"No such folder: {args.videos}")

    videos = sorted(
        os.path.join(args.videos, f)
        for f in os.listdir(args.videos)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    )
    if not videos:
        sys.exit(f"No videos found in {args.videos}")

    print(f"\nSession '{args.session}' — {len(videos)} video(s), sampling {args.fps}/s\n")
    os.makedirs(args.out, exist_ok=True)

    all_rows = []
    for v in videos:
        all_rows.extend(
            extract_one(v, args.out, args.session, args.fps, args.blur_drop, args.max_side)
        )

    if not all_rows:
        sys.exit("\nNo frames written.")

    # Append to metadata.csv so repeat sessions accumulate into one dataset.
    meta_path = os.path.join(args.out, "metadata.csv")
    fields = ["file_name", "label", "session", "source_video", "timestamp_s", "sharpness"]
    exists = os.path.isfile(meta_path)
    with open(meta_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(all_rows)

    # ---- summary + an honest warning about what you can conclude ----
    from collections import Counter, defaultdict

    per_label = Counter(r["label"] for r in all_rows)
    sessions = defaultdict(set)
    with open(meta_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sessions[r["label"]].add(r["session"])

    print(f"\n  {len(all_rows)} frames written to {args.out}/")
    print(f"  metadata.csv now describes the full dataset\n")
    print(f"  {'class':<16}{'this run':>10}{'sessions':>10}")
    for lab in sorted(per_label):
        print(f"  {lab:<16}{per_label[lab]:>10}{len(sessions[lab]):>10}")

    thin = [l for l in sessions if len(sessions[l]) < 2]
    if thin:
        print(
            "\n  WARNING: these classes have only ONE capture session:\n"
            f"    {', '.join(sorted(thin))}\n"
            "  Frames from a single video are near-duplicates of each other, so any\n"
            "  accuracy you measure now is inflated and does not predict real use.\n"
            "  Capture each room again on a different day before you trust a number."
        )
    if "other" not in sessions:
        print(
            "\n  WARNING: no 'other' class. Without one, the model must assign every\n"
            "  corridor, stairwell and blank wall to one of your rooms. Record a walk\n"
            "  through the in-between spaces and save it as other.mp4."
        )
    print()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# If OpenCV cannot open an iPhone video (HEVC), convert it first:
#     brew install ffmpeg
#     ffmpeg -i IMG_1234.MOV -c:v libx264 -crf 20 videos/library.mp4
# ---------------------------------------------------------------------------
