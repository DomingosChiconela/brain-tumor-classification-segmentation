"""
alexnet_hypermodel.py

Espaço de busca (hiperparâmetros) específico do AlexNet, para uso com
keras_tuner.GridSearch. Sabe traduzir os hp recebidos do tuner nos
kwargs esperados por build_alexnet — só este ficheiro conhece essa
correspondência; o resto do grid_search é agnóstico ao modelo.

Segue a mesma filosofia de models/alexnet.py: cada arquitectura tem o
seu próprio ficheiro de builder, e agora também o seu próprio ficheiro
de espaço de busca. Ao criar uma nova arquitectura (ex: ResNet), cria-se
um resnet_hypermodel.py análogo a este.
"""

import keras_tuner as kt
import tensorflow as tf

from src.classification.models.alexnet import build_alexnet


class AlexNetHyperModel(kt.HyperModel):
    """
    HyperModel do AlexNet para keras_tuner.GridSearch.

    Parameters
    ----------
    num_classes : int
        Número de classes de saída.
    search_space : dict
        Hiperparâmetros a testar. As chaves têm de corresponder aos
        parâmetros de build_alexnet. Exemplo:
            {
                "dropout": [0.3, 0.5],
                "conv_filters": [
                    [64, 128, 256, 256, 128],
                    [96, 256, 384, 384, 256],
                ],
                "dense_units": [[2048, 2048], [4096, 4096]],
            }
    lr_choices : list[float]
        Learning rates a testar, ex: [0.001, 0.0001].
    """

    # Parâmetros cujo valor é uma lista/tupla (ex: conv_filters,
    # kernel_sizes, dense_units). Precisam de ser serializados para
    # string antes de ir para hp.Choice, porque este só aceita tipos
    # primitivos (str, int, float, bool), e desserializados de volta
    # ao construir o modelo.
    LIST_PARAMS = {"conv_filters", "kernel_sizes", "dense_units"}

    def __init__(self, num_classes: int, search_space: dict, lr_choices: list):
        self.num_classes = num_classes
        self.search_space = search_space
        self.lr_choices = lr_choices

    def build(self, hp):
        # Liberta o grafo/tensores do trial ANTERIOR antes de construir
        # um modelo novo. Sem isto, o keras_tuner acumula memoria de GPU
        # a cada trial (o TensorFlow nao liberta automaticamente), o que
        # ao fim de varios trials causa corrupcao no BFC allocator da GPU
        # (erro tipico: "Dst tensor is not initialized" seguido de
        # "Check failed: c->in_use()..."). E um problema conhecido do
        # keras_tuner em GPU, nao um bug do teu codigo.
        tf.keras.backend.clear_session()

        model_kwargs = {}

        for param_name, values in self.search_space.items():
            if param_name in self.LIST_PARAMS:
                choice_str = hp.Choice(param_name, values=[str(v) for v in values])
                model_kwargs[param_name] = eval(choice_str)
            else:
                model_kwargs[param_name] = hp.Choice(param_name, values=values)

        model = build_alexnet(num_classes=self.num_classes, **model_kwargs)

        lr = hp.Choice("lr", values=self.lr_choices)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model