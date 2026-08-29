"""
Espaço de busca específico do ResNet. Estrutura igual ao AlexNet, mas
com os parâmetros que fizerem sentido para essa arquitectura
(ex: num_blocks_per_stage, filters_per_stage, use_bottleneck).
"""
import keras_tuner as kt
import tensorflow as tf

# from models.resnet import build_resnet


class ResNetHyperModel(kt.HyperModel):
    LIST_PARAMS = {"filters_per_stage", "blocks_per_stage"}

    def __init__(self, num_classes: int, search_space: dict, lr_choices: list):
        self.num_classes = num_classes
        self.search_space = search_space
        self.lr_choices = lr_choices

    def build(self, hp):
        raise NotImplementedError("Activar quando build_resnet existir.")