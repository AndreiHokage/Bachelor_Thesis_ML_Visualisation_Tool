
class GeneratorRealismFactoryInterface:

    def __init__(self, list_prototype_signs):
        self.__list_prototype_signs = list_prototype_signs

    def get_list_prototype_signs(self):
        return self.__list_prototype_signs

    def createFactory(self):
        pass