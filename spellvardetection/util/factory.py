import inspect
import typing

from spellvardetection.lib.util import load_from_file_if_string

## get all subclass
def all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]

class Factory:

    def __init__(self):
        self.factories_for_classes = dict()
        self.classes_for_name = dict()

    def add_object_hierarchy(self, type_name,  base_cls, create_func=None):

        ## test that type_name and base_cls are not already registered
        if type_name in self.classes_for_name:
            raise ValueError('There is already a factory method for ' + type_name + '.')

        factory_objects = dict()

        for cls in all_subclasses(base_cls):

            name = getattr(cls, 'name', None)
            if not inspect.isabstract(cls) and name is not None:
                if not (create_func is not None and hasattr(cls, create_func)):
                    init_sig = inspect.signature(cls.__init__)
                else:
                    init_sig = inspect.signature(getattr(cls, create_func))

                def object_initializer(cls):
                    if create_func is not None and hasattr(cls, create_func):
                        return lambda options: getattr(cls, create_func)(**options)
                    else:
                        return lambda options: cls(**options)

                factory_info = {
                    ## get required arguments (those that have no default value)
                    "required": [param.name for param in init_sig.parameters.values() if param.name != 'self' and param.default is inspect.Parameter.empty],
                    ## get annotations for arguments
                    "annotations": {param.name: param.annotation for param in init_sig.parameters.values() if param.name != 'self'},
                    ## function to instantiate generator with given options
                    "initializer": object_initializer(cls)
                }

                factory_objects[name] = factory_info


        def factory(object_options):

            ## object_options are either a dict or encoded as json (possibly in a file)
            object_options = load_from_file_if_string(object_options)

            if 'type' not in object_options:
                raise ValueError('Type of object needs to be given')

            object_type = object_options['type']

            options = object_options.get('options', {})


            if object_type not in factory_objects:
                raise ValueError('No candidate ' + type_name + ' of type "' + object_type + '" exists.')

            missing_options = [name for name in factory_objects[object_type]["required"] if name not in options]
            if missing_options:
                raise ValueError('Missing options [' + ', '.join(missing_options) + '] for ' + type_name + ' of type ' + object_type)

            # create objects that are needed as arguments
            for option in options.keys():
                if options[option] is not None:
                    if factory_objects[object_type]["annotations"][option] in self.factories_for_classes and options[option] and not isinstance(options[option], factory_objects[object_type]["annotations"][option]):
                        options[option] = self.create_from_cls(factory_objects[object_type]["annotations"][option], options[option])
                    ## handle sequences of registered types
                    elif hasattr(factory_objects[object_type]["annotations"][option], '__origin__') and factory_objects[object_type]["annotations"][option].__origin__ is typing.Sequence:
                        ## only handle sequences with objects of the same type
                        if len(factory_objects[object_type]["annotations"][option].__args__) == 1:
                            the_type = factory_objects[object_type]["annotations"][option].__args__[0]
                            if the_type in self.factories_for_classes:
                                options[option] = [self.create_from_cls(the_type, opt) for opt in options[option]]

            return factory_objects[object_type]["initializer"](options)

        self.classes_for_name[type_name] = base_cls
        self.add_factory_method(base_cls, factory)

    def add_factory_method(self, cls, method):

        if cls in self.factories_for_classes:
            raise ValueError('There is already a factory method for ' + cls.__name__ + '.')

        self.factories_for_classes[cls] = method

    def create_from_cls(self, cls, options):

        if cls not in self.factories_for_classes:
            raise ValueError('There is no factory method for ' + cls.__name__ + '.')

        return self.factories_for_classes[cls](options)

    def create_from_name(self, type_name, options):

        if type_name not in self.classes_for_name:
            raise ValueError('There is no factory method for ' + type_name + '.')

        class_name = self.classes_for_name[type_name]
        return self.create_from_cls(class_name, options)



factory = Factory()
