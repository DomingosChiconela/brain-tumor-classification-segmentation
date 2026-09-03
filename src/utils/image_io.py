"""
image_io.py

Utilitário partilhado de leitura/redimensionamento de imagens, usado
por build_dataset.py (classificação) e build_dataset_segmentation.py
(segmentação), para não duplicar a lógica de abrir + converter +
redimensionar em dois scripts.

Mantém-se deliberadamente pequeno e sem dependências de TensorFlow —
só PIL/numpy — porque corre em scripts de pré-processamento standalone,
não dentro do grafo de treino.
"""

from pathlib import Path

import numpy as np
from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_and_resize_image(
    path: Path,
    size: tuple,
    channels: int = 3,
    resample: int = Image.BILINEAR,
) -> np.ndarray:
    """
    Abre uma imagem, converte para RGB ou escala de cinzentos, e
    redimensiona. Devolve um array uint8 com shape (H, W, channels).

    Parameters
    ----------
    path : Path
        Caminho da imagem.
    size : tuple
        (largura, altura) — mesma convenção que Image.resize.
    channels : int
        3 para RGB, 1 para escala de cinzentos.
    resample : int
        Filtro de reamostragem do PIL. Usar Image.BILINEAR para imagens
        normais (suaviza), e Image.NEAREST para máscaras binárias
        (preserva os valores exactos, sem criar tons intermédios).
    """
    img = Image.open(path)
    img = img.convert("RGB") if channels == 3 else img.convert("L")
    img = img.resize(size, resample)
    arr = np.array(img, dtype=np.uint8)
    if channels == 1:
        arr = arr[..., np.newaxis]
    return arr


def collect_image_paths(directory: Path) -> list:
    """Lista, ordenada e determinística, dos ficheiros de imagem válidos numa pasta."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )