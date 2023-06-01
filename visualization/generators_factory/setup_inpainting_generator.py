import json
import os

from ML_Traffic_Visualization_Tool.settings import BASE_DIR
from visualization.generators_factory.GeneralEntireGeneratorRealismFactory import GeneralEntireGeneratorRealismFactory
from visualization.generators_factory.UnetPastedGeneratorInpaintingFactory import UnetPastedGeneratorInpaintingFactory


def setup_inpainting_generator_environment(id_generator, data):
    generatorFactory = None
    if id_generator == 'unet_pasted_inpainting_generator':
        create_path = data[id_generator]['create_path']
        image_size = data[id_generator]['image_size']
        print(create_path)
        generatorFactory = UnetPastedGeneratorInpaintingFactory(id_generator, create_path, image_size)
    return generatorFactory

# print(os.path.join(BASE_DIR, r"productionfiles/visualization/InpaintingGeneratorConfig.json"))
# with open(os.path.join(BASE_DIR, r"productionfiles\visualization\InpaintingGeneratorConfig.json")) as f:
#     data = json.load(f)
# factory = setup_inpainting_generator_environment('unet_pasted_inpainting_generator', data)
# generator = factory.createFactory()
# generator.summary()