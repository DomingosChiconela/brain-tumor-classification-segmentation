"""
build_dataset.py

Reads a folder structure like:

    DATASET/classification/
        Training/
            glioma/
            meningioma/
            notumor/
            pituitary/
        Testing/
            glioma/
            meningioma/
            notumor/
            pituitary/

where the class label for each image is implicit in its parent folder name
(no CSV needed), and produces a single compressed .npz file containing:

    X_train, y_train, X_test, y_test, class_names

Usage:
    python build_dataset.py \
        --dataset_dir dataset/classification \
        --img_size 224 224 \
        --channels 3 \
        --output brain_tumor_dataset.npz

Then load it anywhere with:
    data = np.load("brain_tumor_dataset.npz", allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from utils.constants import IMAGE_SIZE

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_image_paths(class_dir: Path):
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def load_split(split_dir: Path, img_size, channels, class_to_idx):
    """
    Loads every image under split_dir/<class_name>/*.ext into a stacked
    numpy array, assigning labels from class_to_idx (built from the
    Training split so Training/Testing always share the same mapping).
    """
    images, labels = [], []

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name not in class_to_idx:
            print(f"  [warn] '{class_name}' not seen in Training, skipping")
            continue
        label = class_to_idx[class_name]

        image_paths = collect_image_paths(class_dir)
        print(f"  {class_name:<12} -> {len(image_paths):5d} images (label {label})")

        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                img = img.convert("RGB") if channels == 3 else img.convert("L")
                img = img.resize(img_size, Image.BILINEAR)
                arr = np.array(img, dtype=np.uint8)
                if channels == 1:
                    arr = arr[..., np.newaxis]  # keep a channel dim for TF
                images.append(arr)
                labels.append(label)
            except Exception as e:
                print(f"  [skip] {img_path}: {e}")

    X = np.stack(images, axis=0) if images else np.empty((0, *img_size, channels), dtype=np.uint8)
    y = np.array(labels, dtype=np.int64)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset/classification",
                         help="Path containing Training/ and Testing/ folders")
    parser.add_argument("--img_size", type=int, nargs=2, default=[IMAGE_SIZE, IMAGE_SIZE],
                         metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--output", type=str, default="dataset/classification/brain_tumor_dataset.npz")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_dir = dataset_dir / "Training"
    test_dir = dataset_dir / "Testing"
    img_size = tuple(args.img_size)

    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found at {train_dir}")

    # Build the class->index mapping from Training (alphabetical, deterministic)
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    print(f"Classes found: {class_to_idx}\n")

    print("Loading Training split...")
    X_train, y_train = load_split(train_dir, img_size, args.channels, class_to_idx)

    X_test, y_test = np.empty((0, *img_size, args.channels), dtype=np.uint8), np.empty((0,), dtype=np.int64)
    if test_dir.exists():
        print("\nLoading Testing split...")
        X_test, y_test = load_split(test_dir, img_size, args.channels, class_to_idx)
    else:
        print(f"\n[warn] No Testing folder found at {test_dir}, skipping")

    print(f"\nTrain tensor: {X_train.shape}, labels: {y_train.shape}")
    print(f"Test tensor:  {X_test.shape}, labels: {y_test.shape}")

    # Sanity check: class balance
    print("\nClass balance (train):")
    for name, idx in class_to_idx.items():
        count = int((y_train == idx).sum())
        print(f"  {name:<12} {count}")

    np.savez_compressed(
        args.output,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        class_names=np.array(class_names),
    )
    print(f"\nSaved everything to {args.output}")


if __name__ == "__main__":
    main()