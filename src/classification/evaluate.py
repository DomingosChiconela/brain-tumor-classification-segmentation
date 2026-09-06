"""
evaluate.py

Script de avaliação genérico para modelos de CLASSIFICAÇÃO de tumores
cerebrais. Carrega um checkpoint já treinado (.keras) e avalia-o no
conjunto de TESTE (nunca visto durante o treino nem a validação).

Reutiliza load_datasets.py e preprocessing.py — a normalização aplicada
ao X_test aqui é EXACTAMENTE a mesma usada no treino, por isso as
métricas reflectem o desempenho real do modelo e não um artefacto de
pré-processamento diferente entre treino e avaliação.

Uso:
    python evaluate.py --checkpoint checkpoints/classification/alexnet_best_model.keras
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.load_datasets import load_classification_dataset
from src.utils.preprocessing import normalize_images

REPORTS_DIR = Path("reports")


def evaluate_model(checkpoint_path: str, reports_dir: Path = REPORTS_DIR):
    """
    Avalia um modelo treinado no conjunto de teste.

    Parameters
    ----------
    checkpoint_path : str
        Caminho para o ficheiro .keras do modelo treinado.
    reports_dir : Path
        Pasta onde a matriz de confusão (.png) e o relatório (.json)
        são guardados.

    Returns
    -------
    dict
        Relatório de classificação (precision/recall/f1 por classe).
    """
    checkpoint_path = Path(checkpoint_path)
    model_name = checkpoint_path.stem  # ex: "alexnet_best_model"

    print(f"A carregar modelo: {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path)

    _, _, X_test, y_test, class_names = load_classification_dataset()
    X_test = normalize_images(X_test)
    class_names = [str(c) for c in class_names]

    print(f"Test: {X_test.shape}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n--- Classification Report ---")
    report_text = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0,
    )
    print(report_text)

    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)

    reports_dir.mkdir(parents=True, exist_ok=True)

    # Guardar matriz de confusão como imagem
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão — {model_name}")
    plt.tight_layout()
    cm_path = reports_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nMatriz de confusão guardada em: {cm_path}")

    # Guardar relatório em JSON (útil para comparar modelos depois)
    report_path = reports_dir / f"{model_name}_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Relatório guardado em: {report_path}")

    return report_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação de modelos de classificação de tumores cerebrais")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Caminho para o ficheiro .keras do modelo treinado")
    args = parser.parse_args()

    evaluate_model(args.checkpoint)