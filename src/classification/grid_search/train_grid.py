"""
train_grid.py

CLI para correr grid search (keras_tuner.GridSearch) sobre qualquer
modelo registado em GRID_HYPERMODEL_REGISTRY. Cada arquitectura tem o
seu próprio HyperModel (ver grid_search/hypermodels/), este ficheiro
só faz o dispatch — tal como train.py faz com MODEL_REGISTRY.

Uso:
    python train_grid.py --model alexnet --epochs 15 \
        --search_space '{"dropout": [0.3, 0.5], "conv_filters": [[64,128,256,256,128],[96,256,384,384,256]]}' \
        --lr_choices '[0.001, 0.0001]'

    python train_grid.py --model resnet --epochs 15 \
        --search_space '{"filters_per_stage": [[64,128,256],[32,64,128]]}' \
        --lr_choices '[0.001]'

As chaves de --search_space têm de corresponder aos parâmetros do
build_* da arquitectura escolhida (ver o HyperModel correspondente
para a lista de parâmetros esperados).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import keras_tuner as kt
import tensorflow as tf

from src.classification.grid_search.hypermodels.alexnet_hypermodel import AlexNetHyperModel
# Descomentar quando build_resnet existir:
from src.classification.grid_search.hypermodels.resnet_hypermodel import ResNetHyperModel
from src.classification.grid_search.grid_tracking_bridge import bridge_tuner_results
from src.utils.load_datasets import load_classification_dataset
from src.utils.preprocessing import normalize_images, split_train_val
from src.utils.paths import build_path

# Cada modelo tem o seu próprio HyperModel, tal como MODEL_REGISTRY em
# train.py associa cada modelo ao seu build_*. A lógica de espaço de
# busca muda por arquitectura; este é o único sítio onde essa escolha
# é feita.
GRID_HYPERMODEL_REGISTRY = {
    "alexnet": AlexNetHyperModel,
    "resnet": ResNetHyperModel,
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


def run_grid_search(
    model_name: str,
    search_space: dict,
    lr_choices: list,
    epochs: int = 15,
    batch_size: int = 32,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    experiments_dir: Path = EXPERIMENTS_DIR,
):
    if model_name not in GRID_HYPERMODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{model_name}' não tem HyperModel de grid search registado. "
            f"Opções: {list(GRID_HYPERMODEL_REGISTRY)}"
        )

    grid_id = "grid_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Grid '{grid_id}' para '{model_name}'.")
    print(f"Espaço de busca: {search_space}")
    print(f"Learning rates: {lr_choices}")

    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_data()

    hypermodel_cls = GRID_HYPERMODEL_REGISTRY[model_name]
    hypermodel = hypermodel_cls(
        num_classes=len(class_names),
        search_space=search_space,
        lr_choices=lr_choices,
    )

    tuner_dir = checkpoint_dir / "grid_search" / model_name
    tuner = kt.GridSearch(
        hypermodel,
        objective="val_accuracy",
        directory=str(tuner_dir),
        project_name=grid_id,
        overwrite=True,
    )

    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True,
            ),
        ],
    )

    leaderboard = bridge_tuner_results(
        tuner=tuner,
        model_name=model_name,
        grid_id=grid_id,
        checkpoint_dir=checkpoint_dir,
        experiments_dir=experiments_dir,
    )

    return tuner, leaderboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid search (keras_tuner) para modelos de classificação de tumores cerebrais"
    )
    parser.add_argument("--model", choices=GRID_HYPERMODEL_REGISTRY.keys(), required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--search_space", type=str, required=True,
        help=(
            "Espaço de busca da ARQUITECTURA em formato JSON. As chaves têm "
            "de corresponder aos parâmetros do build_* do modelo escolhido. "
            'Ex: \'{"dropout": [0.3, 0.5], "conv_filters": [[64,128,256,256,128]]}\''
        ),
    )
    parser.add_argument(
        "--lr_choices", type=str, default="[0.0001]",
        help='Learning rates a testar, em JSON. Ex: \'[0.001, 0.0001]\'',
    )
    args = parser.parse_args()

    try:
        search_space = json.loads(args.search_space)
    except json.JSONDecodeError as e:
        raise ValueError(f"--search_space não é um JSON válido: {e}")

    try:
        lr_choices = json.loads(args.lr_choices)
    except json.JSONDecodeError as e:
        raise ValueError(f"--lr_choices não é um JSON válido: {e}")

    run_grid_search(
        model_name=args.model,
        search_space=search_space,
        lr_choices=lr_choices,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )