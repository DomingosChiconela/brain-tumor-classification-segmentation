from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization,
    Activation, Concatenate, Dropout, Resizing,
)

from src.utils.constants import IMAGE_SIZE


def conv_block(x, filters: int, kernel_size: int, batch_norm: bool, name_prefix: str):
    """
    Bloco convolucional padrão do U-Net: 2x (Conv2D -> [BatchNorm] -> ReLU).
    Usado tanto no encoder como no decoder — mesma lógica, filtros diferentes.
    """
    for i in range(2):
        x = Conv2D(
            filters, kernel_size, padding="same",
            kernel_initializer="he_normal",
            name=f"{name_prefix}_conv{i + 1}",
        )(x)
        if batch_norm:
            x = BatchNormalization(name=f"{name_prefix}_bn{i + 1}")(x)
        x = Activation("relu", name=f"{name_prefix}_relu{i + 1}")(x)
    return x


def build_unet(
    input_shape: tuple = (IMAGE_SIZE, IMAGE_SIZE, 3),
    num_classes: int = 1,
    base_filters: int = 64,
    depth: int = 4,
    kernel_size: int = 3,
    dropout: float = 0.0,
    batch_norm: bool = True,
) -> Model:
    """
    Constrói um modelo U-Net para segmentação, na API funcional do Keras
    (necessária por causa das skip connections — o U-Net não é uma
    pilha linear como o AlexNet, por isso não pode ser Sequential).

    Parameters
    ----------
    input_shape : tuple
        Formato da imagem de entrada (altura, largura, canais).
        Por omissão (IMAGE_SIZE, IMAGE_SIZE, 3), lido de
        src/utils/constants.py — alinhado com
        build_dataset_segmentation.py. Se IMAGE_SIZE for divisível
        por 2**depth (ex: com depth=4, IMAGE_SIZE=192 -> 192/16=12
        exacto), evita a necessidade da camada Resizing de
        realinhamento descrita em Notes abaixo.
    num_classes : int
        Número de canais de saída. Por omissão 1, para segmentação
        BINÁRIA (tumor vs fundo) — confirmado nas máscaras reais do
        dataset, que só têm 2 valores distintos por pixel. A activação
        da última camada é escolhida automaticamente consoante este
        valor: "sigmoid" se num_classes == 1, "softmax" caso contrário
        (multi-classe, ex: se um dia quiseres segmentar por tipo de
        tumor em vez de binário).
    base_filters : int
        Número de filtros do primeiro nível do encoder. Duplica a cada
        nível (padrão U-Net): 64 -> 128 -> 256 -> 512 -> ...
    depth : int
        Número de níveis de downsampling/upsampling. O U-Net clássico
        usa depth=4. Cada nível reduz a resolução espacial a metade
        via MaxPooling2D(2,2).
    kernel_size : int
        Tamanho do kernel (NxN) usado em todas as convoluções 3x3
        do encoder/decoder (não afecta o Conv2DTranspose, que usa
        sempre kernel 2x2/stride 2 para o upsampling).
    dropout : float
        Taxa de dropout aplicada no bottleneck e em cada bloco do
        decoder, para reduzir overfitting. 0.0 desactiva.
    batch_norm : bool
        Se True, aplica BatchNormalization após cada convolução.
        Recomendado manter True — acelera e estabiliza o treino,
        sobretudo com redes tão profundas quanto um U-Net de depth=4.

    Returns
    -------
    tensorflow.keras.Model
        Modelo U-Net NÃO compilado (sem optimizer/loss definidos).

    Notes
    -----
    Se input_shape não for perfeitamente divisível por 2**depth (ex:
    150 com depth=4), os tensores do decoder e as skip connections
    correspondentes ficam com tamanhos ligeiramente diferentes depois
    do upsampling. Isto é resolvido automaticamente com uma camada
    Resizing que realinha o tensor do decoder ao tamanho exacto da
    skip connection, antes de concatenar — não é preciso redimensionar
    o dataset para uma potência de 2.
    """
    if depth < 1:
        raise ValueError("depth deve ser >= 1.")
    if base_filters < 1:
        raise ValueError("base_filters deve ser >= 1.")
    if num_classes < 1:
        raise ValueError("num_classes deve ser >= 1.")
    if not (0.0 <= dropout < 1.0):
        raise ValueError("dropout deve estar em [0.0, 1.0).")

    inputs = Input(shape=input_shape, name="input_image")
    x = inputs

    # Encoder: guarda a saída de cada nível ANTES do pooling, para
    # ligar ao decoder via skip connection.
    skip_connections = []
    for level in range(depth):
        filters = base_filters * (2 ** level)
        x = conv_block(x, filters, kernel_size, batch_norm, name_prefix=f"enc{level + 1}")
        skip_connections.append(x)
        x = MaxPooling2D(pool_size=(2, 2), name=f"pool{level + 1}")(x)

    # Bottleneck: ponto mais profundo da rede, sem skip connection.
    bottleneck_filters = base_filters * (2 ** depth)
    x = conv_block(x, bottleneck_filters, kernel_size, batch_norm, name_prefix="bottleneck")
    if dropout > 0:
        x = Dropout(dropout, name="bottleneck_dropout")(x)

    # Decoder: percorre os níveis na ordem inversa do encoder,
    # reconstruindo a resolução espacial e fundindo com a skip
    # connection correspondente.
    for level in reversed(range(depth)):
        filters = base_filters * (2 ** level)
        x = Conv2DTranspose(
            filters, kernel_size=2, strides=2, padding="same",
            name=f"upconv{level + 1}",
        )(x)

        skip = skip_connections[level]

        # Realinha o tensor do decoder ao tamanho exacto da skip
        # connection, caso input_shape não seja perfeitamente
        # divisível por 2**depth (ver Notes acima).
        skip_h, skip_w = skip.shape[1], skip.shape[2]
        if x.shape[1] != skip_h or x.shape[2] != skip_w:
            x = Resizing(skip_h, skip_w, interpolation="bilinear",
                         name=f"resize_align{level + 1}")(x)

        x = Concatenate(name=f"concat{level + 1}")([x, skip])
        x = conv_block(x, filters, kernel_size, batch_norm, name_prefix=f"dec{level + 1}")
        if dropout > 0:
            x = Dropout(dropout, name=f"dec{level + 1}_dropout")(x)

    out_activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = Conv2D(
        num_classes, kernel_size=1, activation=out_activation,
        name="segmentation_output",
    )(x)

    return Model(inputs=inputs, outputs=outputs, name="UNet")


if __name__ == "__main__":

    model = build_unet()
    model.summary()