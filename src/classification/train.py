"""
Uso:
    python train.py --model alexnet --epochs 30
    python train.py --model alexnet --epochs 30 --batch_size 16 --lr 0.0005
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from models.alexnet import build_alexnet
# Quando o resnet.py existir: from models.resnet import build_resnet
from utils.load_datasets import load_classification_dataset
from utils.preprocessing import normalize_images, split_train_val

MODEL_REGISTRY = {
    "alexnet": build_alexnet,
    # "resnet": build_resnet,
}

CHECKPOINT_DIR = Path("checkpoints/classification")


def load_data(val_split: float = 0.15, seed: int = 42):
   
    X_train_full, y_train_full, X_test, y_test, class_names = load_classification_dataset()

    X_train_full = normalize_images(X_train_full)
    X_test = normalize_images(X_test)

    X_train, X_val, y_train, y_val = split_train_val(
        X_train_full, y_train_full, val_split=val_split, seed=seed,
    )

    print(f"Classes: {list(class_names)}")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, class_names


def train_model(
    model_name: str,
    model_kwargs: dict = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    checkpoint_dir: Path = CHECKPOINT_DIR,
):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo '{model_name}' não existe. Opções: {list(MODEL_REGISTRY)}")

    model_kwargs = model_kwargs or {}

    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_data()

    build_fn = MODEL_REGISTRY[model_name]
    model = build_fn(num_classes=len(class_names), **model_kwargs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",  # y são inteiros, não one-hot
        metrics=["accuracy"],
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_best_model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    print(f"\nMelhor modelo guardado em: {checkpoint_path}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino de modelos de classificação de tumores cerebrais")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--model_kwargs", type=str, default="{}",
        help=(
            'Hiperparâmetros da ARQUITECTURA em formato JSON. '
            'Ex: \'{"dropout": 0.5, "conv_filters": [64, 128, 256, 256, 128]}\''
        ),
    )
    args = parser.parse_args()

    try:
        model_kwargs = json.loads(args.model_kwargs)
    except json.JSONDecodeError as e:
        raise ValueError(f"--model_kwargs não é um JSON válido: {e}")

    train_model(
        model_name=args.model,
        model_kwargs=model_kwargs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )