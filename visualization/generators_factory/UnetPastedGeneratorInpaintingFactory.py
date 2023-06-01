import os

from ML_Traffic_Visualization_Tool.settings import BASE_DIR
from visualization.augmenting_traffic_signs.networks_inpainting import create_generator_unet
from visualization.generators_factory.GeneratorInpaintingFactoryInterface import GeneratorInpaintingFactoryInterface


class UnetPastedGeneratorInpaintingFactory(GeneratorInpaintingFactoryInterface):

    def __init__(self, id_generator, create_path, image_size):
        super(GeneratorInpaintingFactoryInterface, self).__init__()
        self.__id_generator = id_generator
        self.__create_path = create_path
        self.__image_size = image_size

    def createFactory(self):
        generator = create_generator_unet(self.__image_size)
        generator.load_weights(os.path.join(BASE_DIR, self.__create_path))
        return generator
