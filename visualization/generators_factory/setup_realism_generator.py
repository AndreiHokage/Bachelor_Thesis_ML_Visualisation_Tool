import json
import os

from ML_Traffic_Visualization_Tool.settings import BASE_DIR
from visualization.generators_factory.GeneralEntireGeneratorRealismFactory import GeneralEntireGeneratorRealismFactory


def setup_realism_generator_environment(id_generator, data):
    generatorFactory = None
    if id_generator == 'pasted_realism_generator':
        model_path = data[id_generator]['model_path']
        list_prototype_signs = data[id_generator]['list_prototype_signs']
        generatorFactory = GeneralEntireGeneratorRealismFactory(id_generator, list_prototype_signs, model_path)
    elif id_generator == 'pasted_realism_generator_78':
        model_path = data[id_generator]['model_path']
        list_prototype_signs = data[id_generator]['list_prototype_signs']
        generatorFactory = GeneralEntireGeneratorRealismFactory(id_generator, list_prototype_signs, model_path)
    return generatorFactory

# with open(os.path.join(BASE_DIR, r"productionfiles\visualization\RealismGeneratorConfig.json")) as f:
#     data = json.load(f)
# factory = setup_realism_generator_environment('pasted_realism_generator', data)
# generator = factory.createFactory()
# print(factory.get_list_prototype_signs())
# generator.summary()