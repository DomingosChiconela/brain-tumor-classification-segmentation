import tensorflow
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add, GlobalAveragePooling2D, Dense, Conv2D, 
    Input, BatchNormalization, Activation, MaxPooling2D, Dropout
)
from src.utils.constants import IMAGE_SIZE

def resnet_block(x, filters, kernel_size=3, stride=1):
    """
    Bloco ResNet padrão (2x Conv2D 3x3).
    realiza a projecao do shortcut quando stride > 1 ou quantidade de canais muda.
    """
    initializer = tensorflow.keras.initializers.HeNormal()
    x_shortcut = x

    # primeira camada convolucional (resolve o downsampling via stride)
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same',
               kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Segunda Convolution
    x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same',
               kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)

    # matching conexao Shortcut (1x1 Conv quando downsampled or muda canal)
    in_channels = x_shortcut.shape[-1]
    if stride != 1 or in_channels != filters:
        x_shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same',
                            kernel_initializer=initializer)(x_shortcut)
        x_shortcut = BatchNormalization()(x_shortcut)

    # Adiciona  identidade (shortcut) e aplica ReLU
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x


def create_resnet(input_shape: tuple = (IMAGE_SIZE, IMAGE_SIZE, 3), 
                  num_classes: int = 4, 
                  num_blocks_list: list = [3, 4, 6, 3],
                  filters_list: list = [64, 128, 256, 512]) -> Model:
    """
    Cria o ResNet-34 padrão com base na arquitetura especificada.
    """
    initializer = tensorflow.keras.initializers.HeNormal()
    inputs = Input(shape=input_shape)

    # Stem:extracao de padrões iniciais
    x = Conv2D(64, kernel_size=7, strides=2, padding='same',
               kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # camadas residuais (1 a 4)
    for i, num_blocks in enumerate(num_blocks_list):
        filters = filters_list[i]
        for j in range(num_blocks):
            # Downsample spatially at the start of Stage 2, 3, and 4 (stride 2)
            stride = 2 if (j == 0 and i > 0) else 1
            x = resnet_block(x, filters=filters, stride=stride)

    # Classificador
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(x)

    return Model(inputs=inputs, outputs=outputs)