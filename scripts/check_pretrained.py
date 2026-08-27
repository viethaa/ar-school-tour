"""
Baseline: stock ImageNet MobileNetV2.

  python scripts/check_pretrained.py            # ordinary photos — proves the toolchain
  python scripts/check_pretrained.py ./photos/  # school photos — proves the gap

Nothing here trains anything.
"""

import os
import sys
import glob

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions)
from tensorflow.keras.utils import load_img, img_to_array, get_file

try:
    import pillow_heif
    pillow_heif.register_heif_opener()          # iPhone .HEIC
except ImportError:
    pass

IMG_SIZE = (224, 224)
BASE = "https://storage.googleapis.com/download.tensorflow.org/example_images/"
SAMPLES = {
    "sunflower.jpg": BASE + "592px-Red_sunflower.jpg",
    "labrador.jpg": BASE + "YellowLabradorLooking_new.jpg",
    "grace_hopper.jpg": BASE + "grace_hopper.jpg",
}
EXTS = ("jpg", "jpeg", "png", "webp", "heic", "JPG", "JPEG", "PNG", "HEIC")


def predict(model, path, top=5):
    img = load_img(path, target_size=IMG_SIZE)
    batch = preprocess_input(np.expand_dims(img_to_array(img), axis=0))
    decoded = decode_predictions(model.predict(batch, verbose=0), top=top)[0]
    print(os.path.basename(path))
    for _id, label, score in decoded:
        print(f"    {label:<24}{score * 100:5.1f}%  {'#' * int(score * 30)}")
    print()


def main():
    model = MobileNetV2(weights="imagenet")
    print(f"MobileNetV2 loaded — {model.count_params():,} parameters\n")

    if len(sys.argv) > 1:
        folder = sys.argv[1]
        if not os.path.isdir(folder):
            sys.exit(f"Not a folder: {folder}")
        paths = sorted(p for e in EXTS
                       for p in glob.glob(os.path.join(folder, "**", f"*.{e}"), recursive=True))
        if not paths:
            sys.exit(f"No images in {folder}")
        print(f"Untrained model on {len(paths)} of your photos.")
        print("Expect object names, not room names. That is the point.\n")
        for p in paths[:20]:
            predict(model, p)
    else:
        for name, url in SAMPLES.items():
            predict(model, get_file(name, url))
        print("Chain works: Python -> TensorFlow -> MobileNetV2 -> image -> prediction")


if __name__ == "__main__":
    main()
