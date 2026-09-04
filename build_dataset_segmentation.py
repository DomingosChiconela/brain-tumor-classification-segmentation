"""
build_dataset_segmentation.py

Lê uma estrutura de pastas como:

    dataset/Segmentation/
        Glioma/
            enh_1841.jpg
            enh_1841_mask.png
            enh_1842.jpg
            enh_1842_mask.png
            ...
        Meningioma/
            ...
        Pituitary tumor/
            ...

onde cada imagem tem uma máscara correspondente na MESMA pasta, com o
mesmo nome base + sufixo "_mask" (a extensão da máscara pode ser
diferente da imagem — o script procura por qualquer extensão válida).

Ao contrário de build_dataset.py (classificação), aqui:
  - não existe uma pasta Training/Testing pré-definida — o split
    treino/teste é feito por este script, de forma ESTRATIFICADA por
    tipo de tumor, para manter a proporção de cada classe em ambos.
  - o alvo (Y) não é um inteiro, é uma máscara binária do mesmo
    tamanho da imagem (0 = fundo, 1 = tumor).
  - o nome da pasta (Glioma/Meningioma/Pituitary tumor) é guardado
    como metadado (tumor_type), útil para estratificar o split e para
    análises posteriores, mas NÃO é o alvo do treino de segmentação.

Produz um único ficheiro .npz comprimido com:

    X_train, Y_train, X_test, Y_test, tumor_type_train, tumor_type_test, class_names

Usage:
    python build_dataset_segmentation.py \
        --dataset_dir dataset/Segmentation \
        --img_size IMAGE_SIZE IMAGE_SIZE \
        --channels 3 \
        --test_split 0.15 \
        --output dataset/segmentation/brain_tumor_segmentation_dataset.npz

Depois carrega-se em qualquer lado com:
    data = np.load("brain_tumor_segmentation_dataset.npz", allow_pickle=True)
    X_train, Y_train = data["X_train"], data["Y_train"]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src.utils.image_io import VALID_EXTENSIONS, collect_image_paths, load_and_resize_image
from  src.utils.constants import IMAGE_SIZE

MASK_BINARY_THRESHOLD = 129  # confirmado via inspect_mask.py: fundo=3, tumor=255


def find_mask_for(image_path: Path, class_dir: Path) -> Path | None:
    """
    Procura o ficheiro de máscara correspondente a image_path, dentro
    da mesma pasta, testando todas as extensões válidas — porque a
    extensão da máscara pode não coincidir com a da imagem original.
    """
    mask_stem = f"{image_path.stem}_mask"
    for ext in VALID_EXTENSIONS:
        candidate = class_dir / f"{mask_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_pairs(class_dir: Path) -> list:
    """
    Devolve a lista de pares (image_path, mask_path) desta pasta de
    classe, ignorando os próprios ficheiros de máscara (que terminam
    em "_mask") ao procurar imagens-base.
    """
    all_files = collect_image_paths(class_dir)
    image_files = [p for p in all_files if not p.stem.endswith("_mask")]

    pairs = []
    for img_path in image_files:
        mask_path = find_mask_for(img_path, class_dir)
        if mask_path is None:
            print(f"  [warn] sem máscara para {img_path.name}, a ignorar")
            continue
        pairs.append((img_path, mask_path))

    return pairs


def load_pairs(pairs: list, img_size: tuple, channels: int) -> tuple:
    """
    Carrega as imagens (BILINEAR, preserva detalhe) e as máscaras
    correspondentes (NEAREST, preserva binário), e binariza as máscaras
    por threshold.
    """
    images, masks = [], []

    for img_path, mask_path in pairs:
        try:
            img_arr = load_and_resize_image(img_path, img_size, channels, resample=Image.BILINEAR)

            mask_arr = load_and_resize_image(mask_path, img_size, channels=1, resample=Image.NEAREST)
            mask_arr = (mask_arr > MASK_BINARY_THRESHOLD).astype(np.uint8)

            images.append(img_arr)
            masks.append(mask_arr)
        except Exception as e:
            print(f"  [skip] {img_path.name}: {e}")

    X = np.stack(images, axis=0) if images else np.empty((0, *img_size, channels), dtype=np.uint8)
    Y = np.stack(masks, axis=0) if masks else np.empty((0, *img_size, 1), dtype=np.uint8)
    return X, Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset/Segmentation",
                         help="Pasta contendo uma subpasta por tipo de tumor (Glioma, Meningioma, ...)")
    parser.add_argument("--img_size", type=int, nargs=2, default=[IMAGE_SIZE, IMAGE_SIZE],
                         metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3,
                         help="Canais da IMAGEM. A máscara é sempre carregada como 1 canal binário.")
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                         default="dataset/segmentation/brain_tumor_segmentation_dataset.npz")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    img_size = tuple(args.img_size)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {dataset_dir}")

    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    print(f"Tipos de tumor encontrados: {class_names}\n")

    # 1) Recolher todos os pares (imagem, máscara) de todas as classes,
    #    guardando o tumor_type de cada par — necessário para o split
    #    estratificado, feito ANTES de carregar as imagens para memória
    #    (evita carregar tudo duas vezes).
    all_pairs = []
    all_tumor_types = []
    for class_dir in class_dirs:
        tumor_type = class_dir.name
        pairs = collect_pairs(class_dir)
        print(f"  {tumor_type:<20} -> {len(pairs)} pares imagem/máscara")
        all_pairs.extend(pairs)
        all_tumor_types.extend([tumor_type] * len(pairs))

    if not all_pairs:
        raise RuntimeError("Nenhum par imagem/máscara encontrado. Verifica --dataset_dir.")

    # 2) Split treino/teste ESTRATIFICADO por tumor_type, para que a
    #    proporção de Glioma/Meningioma/Pituitary seja semelhante em
    #    ambos os conjuntos.
    indices = np.arange(len(all_pairs))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_split,
        random_state=args.seed,
        stratify=all_tumor_types,
    )

    print(f"\nTotal de pares: {len(all_pairs)}")
    print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")

    # 3) Carregar imagens/máscaras só depois do split, e só as
    #    necessárias a cada conjunto.
    print("\nA carregar imagens de treino...")
    train_pairs = [all_pairs[i] for i in train_idx]
    X_train, Y_train = load_pairs(train_pairs, img_size, args.channels)
    tumor_type_train = np.array([all_tumor_types[i] for i in train_idx])

    print("A carregar imagens de teste...")
    test_pairs = [all_pairs[i] for i in test_idx]
    X_test, Y_test = load_pairs(test_pairs, img_size, args.channels)
    tumor_type_test = np.array([all_tumor_types[i] for i in test_idx])

    print(f"\nX_train: {X_train.shape} | Y_train: {Y_train.shape}")
    print(f"X_test:  {X_test.shape} | Y_test:  {Y_test.shape}")

    # Sanity check: proporção de pixels de tumor vs fundo (ajuda a
    # perceber se as máscaras estão desequilibradas — comum em
    # segmentação, e relevante para escolher a loss depois, ex: Dice
    # loss em vez de BCE simples se o desequilíbrio for grande).
    tumor_pixel_ratio = Y_train.mean()
    print(f"\nProporção de pixels de tumor no treino: {tumor_pixel_ratio:.4f}")

    print("\nBalanço por tipo de tumor (treino):")
    for name in class_names:
        count = int((tumor_type_train == name).sum())
        print(f"  {name:<20} {count}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X_train=X_train, Y_train=Y_train,
        X_test=X_test, Y_test=Y_test,
        tumor_type_train=tumor_type_train, tumor_type_test=tumor_type_test,
        class_names=np.array(class_names),
    )
    print(f"\nGuardado em {args.output}")


if __name__ == "__main__":
    main()