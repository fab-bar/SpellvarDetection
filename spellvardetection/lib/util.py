# -*- coding: utf-8 -*-

import json

def load_from_file_if_string(option):
    if isinstance(option, str):
        return json.load(open(option, 'r'))
    else:
        return option

def createFactory(factory_type, object_types):

    def factory(object_type, options):

        if object_type not in object_types:
            raise ValueError('No candidate ' + factory_type + ' of type "' + object_type + '" exists.')

        if not all(name in options for name in object_types[object_type][0]):
            raise ValueError('Missing options for ' + factory_type + ' of type ' + object_type)

        return object_types[object_type][1](options)

    return factory
