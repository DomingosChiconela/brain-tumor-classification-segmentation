"""

Uso:
    python train.py --model alexnet --epochs 30
    python train.py --model alexnet --epochs 30 --batch_size 16 --lr 0.0005
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import tensorflow as tf

from .models.alexnet import build_alexnet
from .models.resnet import create_resnet
from src.utils.load_datasets import load_classification_dataset
from src.utils.preprocessing import normalize_images, split_train_val
from  src.utils.paths  import build_path
from src.utils.experiment_tracking import (
    make_run_id, get_checkpoint_run_dir, save_run_config,
    append_to_runs_log, update_best_if_needed,
)

MODEL_REGISTRY = {
    "alexnet": build_alexnet,
     "resnet": create_resnet,
}

CHECKPOINT_DIR = build_path("checkpoints", "classification")
EXPERIMENTS_DIR = build_path("experiments", "classification")


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
    experiments_dir: Path = EXPERIMENTS_DIR,
):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo '{model_name}' não existe. Opções: {list(MODEL_REGISTRY)}")

    model_kwargs = model_kwargs or {}

    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_data()

    build_fn = MODEL_REGISTRY[model_name]
    model = build_fn(num_classes=len(class_names), **model_kwargs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",  
        metrics=["accuracy"],
    )

   
    run_id = make_run_id()
    run_dir = get_checkpoint_run_dir(checkpoint_dir, model_name, run_id)
    checkpoint_path = run_dir / "model.keras"

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

    # Guardar hiperparâmetros + métricas desta execução, para nunca mais
    # perderes o registo de "o que foi usado para chegar a esta accuracy".
    config = {
        "model_name": model_name,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model_kwargs": model_kwargs,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "best_val_accuracy": max(history.history["val_accuracy"]),
        "best_val_loss": min(history.history["val_loss"]),
        "checkpoint_path": str(checkpoint_path),
    }
    save_run_config(experiments_dir, model_name, run_id, config)
    append_to_runs_log(experiments_dir, config)

    is_new_best = update_best_if_needed(
        checkpoint_dir, experiments_dir, model_name, checkpoint_path,
        config["best_val_accuracy"],
    )

    print(f"\nCheckpoint desta execução: {checkpoint_path}")
    if is_new_best:
        print(f"Novo MELHOR modelo de sempre para '{model_name}'! "
              f"-> {checkpoint_dir / model_name / 'best.keras'}")
    else:
        print(f"Esta execução não superou o melhor anterior "
              f"(val_accuracy={config['best_val_accuracy']:.4f}).")

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