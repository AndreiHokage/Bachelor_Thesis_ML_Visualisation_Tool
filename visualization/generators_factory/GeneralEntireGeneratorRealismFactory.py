import os

import tensorflow as tf
import tensorflow.keras as tfkeras

from ML_Traffic_Visualization_Tool.settings import BASE_DIR
from visualization.generators_factory.GeneratorRealismFactoryInterface import GeneratorRealismFactoryInterface


class GeneralEntireGeneratorRealismFactory(GeneratorRealismFactoryInterface):
    def __init__(self, id_generator, list_prototype_signs, model_path):
        super().__init__(list_prototype_signs)
        self.__id_generator = id_generator
        self.__model_path = model_path

    def createFactory(self):
        generator = tfkeras.models.load_model(os.path.join(BASE_DIR, self.__model_path))
        return generator