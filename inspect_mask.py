"""
inspect_mask.py

Diagnóstico rápido para confirmar, ANTES de correr
build_dataset_segmentation.py em massa, como as tuas máscaras estão
codificadas: binário puro (0/255), binário com anti-aliasing nas
bordas (tons intermédios), ou multi-classe (mais do que 2 valores
distintos, ex: cada tipo de tumor com um cinzento diferente).

Isto determina se MASK_BINARY_THRESHOLD = 127 é apropriado, se precisa
de ajuste, ou se a máscara nem sequer deve ser tratada como binária.

Usage:
    python inspect_mask.py "dataset/Segmentation/Glioma/enh_1841_mask.png"
    python inspect_mask.py "dataset/Segmentation/Glioma/enh_1841_mask.png" --show
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def inspect_mask(path: Path, show: bool = False):
    img = Image.open(path)
    print(f"Ficheiro: {path}")
    print(f"Modo original PIL: {img.mode}")  # 'L', 'RGB', 'P', '1', etc.
    print(f"Tamanho: {img.size}")

    # Analisar em escala de cinzentos, tal como o build_dataset_segmentation faz.
    gray = np.array(img.convert("L"))

    unique_values, counts = np.unique(gray, return_counts=True)
    n_unique = len(unique_values)

    print(f"\nValores de pixel únicos (escala de cinzentos): {n_unique}")
    if n_unique <= 20:
        for v, c in zip(unique_values, counts):
            pct = 100 * c / gray.size
            print(f"  valor {v:>3}  ->  {c:>8} pixels  ({pct:5.1f}%)")
    else:
        print(f"  min={unique_values.min()}, max={unique_values.max()}")
        print(f"  (mais de 20 valores distintos — não parece binário puro)")

    # Diagnóstico automático
    print("\n--- Diagnóstico ---")
    if n_unique == 2 and set(unique_values.tolist()) <= {0, 255}:
        print("Máscara é BINÁRIA PURA (só 0 e 255). Threshold=127 é seguro e exacto.")
    elif n_unique == 2:
        print(f"Máscara é binária, mas com valores {list(unique_values)} em vez de 0/255.")
        midpoint = int((int(unique_values[0]) + int(unique_values[1])) / 2)
        print(f"Ajusta MASK_BINARY_THRESHOLD para ~{midpoint}.")
    elif n_unique <= 10:
        print(f"Máscara tem {n_unique} valores distintos — pode ser anti-aliasing nas")
        print("bordas (não é grave, threshold ainda funciona) OU multi-classe")
        print("(cada valor representa algo diferente — nesse caso NÃO binarizar,")
        print("tratar como segmentação multi-classe em vez de binária).")
    else:
        print(f"Máscara tem {n_unique} valores distintos — provavelmente anti-aliasing")
        print("suave nas bordas do blob. Um threshold em ~127 continua razoável,")
        print("mas confirma visualmente com --show que a borda fica no sítio certo.")

    if show:
        img.convert("L").show(title=f"Máscara original: {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_path", type=str, help="Caminho para um ficheiro de máscara")
    parser.add_argument("--show", action="store_true", help="Abre a imagem para inspecção visual")
    args = parser.parse_args()

    inspect_mask(Path(args.mask_path), show=args.show)