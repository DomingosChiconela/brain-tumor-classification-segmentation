"""
resnet_hypermodel.py

Espaço de busca específico do ResNet, para uso com keras_tuner.GridSearch.
Faz a ponte entre o espaço de busca do keras_tuner (hp) e create_resnet,
tal como alexnet_hypermodel.py faz para build_alexnet.

create_resnet espera:
    num_blocks_list : list[int]  -> nº de blocos residuais por stage
    filters_list     : list[int] -> nº de filtros por stage
Ambas têm o MESMO comprimento (uma entrada por stage) e devem variar
JUNTAS, não independentemente — testar num_blocks_list e filters_list
como escolhas separadas geraria combinações inválidas (comprimentos
diferentes) ou sem sentido arquitectural. Por isso, no search_space,
cada "opção" é um PAR (num_blocks_list, filters_list) já combinado,
não duas listas separadas — ver exemplo abaixo.
"""

import keras_tuner as kt
import tensorflow as tf

from src.classification.models.resnet import create_resnet


class ResNetHyperModel(kt.HyperModel):
    """
    HyperModel do ResNet para keras_tuner.GridSearch.

    Parameters
    ----------
    num_classes : int
        Número de classes de saída.
    search_space : dict
        Hiperparâmetros a testar. As chaves têm de corresponder aos
        parâmetros de create_resnet. Exemplo:
            {
                "stage_config": [
                    {"num_blocks_list": [2, 2, 2, 2], "filters_list": [64, 128, 256, 512]},
                    {"num_blocks_list": [3, 4, 6, 3], "filters_list": [64, 128, 256, 512]}
                ]
            }
        "stage_config" é tratado de forma especial: cada opção é um par
        já combinado de (num_blocks_list, filters_list), garantindo que
        nunca ficam desalinhados. Qualquer outra chave em search_space
        (ex: um futuro "dropout", se create_resnet passar a aceitá-lo
        como argumento) é tratada como hp.Choice normal.
    lr_choices : list[float]
        Learning rates a testar, ex: [0.001, 0.0001].
    """

    LIST_PARAMS = {"num_blocks_list", "filters_list"}

    def __init__(self, num_classes: int, search_space: dict, lr_choices: list):
        self.num_classes = num_classes
        self.search_space = search_space
        self.lr_choices = lr_choices

    def build(self, hp):
        # Ver comentario equivalente em alexnet_hypermodel.py -- liberta
        # o grafo do trial anterior antes de construir um modelo novo,
        # evitando corrupcao de memoria da GPU ao longo de varios trials.
        tf.keras.backend.clear_session()

        model_kwargs = {}

        # stage_config: par (num_blocks_list, filters_list) escolhido
        # em conjunto, para nunca desalinhar os dois.
        if "stage_config" in self.search_space:
            stage_options = self.search_space["stage_config"]
            choice_str = hp.Choice(
                "stage_config",
                values=[str(opt) for opt in stage_options],
            )
            chosen = eval(choice_str)
            model_kwargs["num_blocks_list"] = chosen["num_blocks_list"]
            model_kwargs["filters_list"] = chosen["filters_list"]

        # Restantes hiperparâmetros escalares/lista simples (ex: um
        # futuro "dropout", se create_resnet passar a aceitá-lo).
        for param_name, values in self.search_space.items():
            if param_name == "stage_config":
                continue
            if param_name in self.LIST_PARAMS:
                choice_str = hp.Choice(param_name, values=[str(v) for v in values])
                model_kwargs[param_name] = eval(choice_str)
            else:
                model_kwargs[param_name] = hp.Choice(param_name, values=values)

        model = create_resnet(num_classes=self.num_classes, **model_kwargs)

        lr = hp.Choice("lr", values=self.lr_choices)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model