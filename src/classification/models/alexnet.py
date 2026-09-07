from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, GlobalAveragePooling2D,MaxPool2D
from  src.utils.constants import IMAGE_SIZE



def build_alexnet(
    input_shape: tuple = (IMAGE_SIZE, IMAGE_SIZE, 3),
    num_classes: int = 4,
    conv_filters: tuple = (96, 256, 384, 384, 256),
    kernel_sizes: tuple = (11, 5, 3, 3, 3),
    dense_units: tuple = (512, 128),
    dropout: float = 0.5,
) -> Sequential:
    """
 
    Parameters
    ----------
    input_shape : tuple
        Formato da imagem de entrada (altura, largura, canais).
        Por omissão (IMAGE_SIZE, IMAGE_SIZE, 3), alinhado com build_dataset.py.
    num_classes : int
        Número de classes de saída. Por omissão 4
        (glioma, meningioma, notumor, pituitary).
    conv_filters : tuple[int, int, int, int, int]
        Número de filtros em cada uma das 5 camadas convolucionais.
    kernel_sizes : tuple[int, int, int, int, int]
        Tamanho do kernel (NxN) em cada uma das 5 camadas convolucionais.
    dense_units : tuple[int, int]
        Número de neurónios nas duas camadas densas (fully connected).
    dropout : float
        Taxa de dropout aplicada após cada camada densa, para reduzir
        overfitting.
 
    Returns
    -------
    tensorflow.keras.models.Sequential
        Modelo AlexNet NÃO compilado (sem optimizer/loss definidos).
    """
    if len(conv_filters) != 5 or len(kernel_sizes) != 5:
        raise ValueError("conv_filters e kernel_sizes devem ter exactamente 5 valores.")
    if len(dense_units) != 2:
        raise ValueError("dense_units deve ter exactamente 2 valores.")
 
    model = Sequential(name="AlexNet")
 
    # Bloco 1
    model.add(Conv2D(conv_filters[0], (kernel_sizes[0], kernel_sizes[0]),strides=(1, 1), padding="same", activation="relu",input_shape=input_shape
    ))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
 
    # Bloco 2
    model.add(Conv2D(conv_filters[1], (kernel_sizes[1], kernel_sizes[1]),padding="same",activation="relu",
    ))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
 
    # Bloco 3, 4, 5 — convoluções sucessivas sem pooling entre elas
    model.add(Conv2D(
        conv_filters[2], (kernel_sizes[2], kernel_sizes[2]),
        padding="same", activation="relu",
    ))
    model.add(Conv2D(
        conv_filters[3], (kernel_sizes[3], kernel_sizes[3]),
        padding="same", activation="relu",
    ))
    model.add(Conv2D(
        conv_filters[4], (kernel_sizes[4], kernel_sizes[4]),
        padding="same", activation="relu",
    ))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
 
    # Classificador (fully connected)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(dense_units[0], activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(dense_units[1], activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation="softmax"))
 
    return model
 
 
if __name__ == "__main__":
   
    model = build_alexnet()
    model.summary()
 