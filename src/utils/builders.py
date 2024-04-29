import importlib


def build_object(type, params, imported_module):
    """
    Function to build pythonic object from strings
    :param type: string of what particular object should be taken
    :param params: Dict of parameters for object
    :param imported_module: module in which type object is present
    :return: Pythonic object imported and built with params
    """
    module = importlib.import_module(imported_module)
    if params:
        return getattr(module, type)(**params)
    else:
        return getattr(module, type)()


def build_loaded_object(type, params, imported_module, path):
    """
    Function to build pythonic object of Lightning module from path
    :param type: string of what particular object should be taken
    :param params: Dict of parameters for object
    :param imported_module: module in which type object is present
    :param path: path to checkpoint load
    :return: Pythonic object imported and built with params
    """
    module = importlib.import_module(imported_module)
    if params:
        return getattr(module, type).load_from_checkpoint(path, **params)
    else:
        return getattr(module, type).load_from_checkpoint(path)
