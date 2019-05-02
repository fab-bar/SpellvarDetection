from spellvardetection.generator import _AbstractCandidateGenerator
from spellvardetection.type_filter import _AbstractTrainableTypeFilter, _AbstractTypeFilter
from spellvardetection.util.feature_extractor import FeatureExtractorMixin

from spellvardetection.lib.util import load_from_file_if_string
from .factory import Factory

def create_base_factory():
    factory = Factory(option_parser=load_from_file_if_string)
    factory.add_factory_method(list, load_from_file_if_string)
    factory.add_factory_method(dict, load_from_file_if_string)
    factory.add_factory_method(set, lambda x: set(load_from_file_if_string(x)))

    factory.add_object_hierarchy("generator", _AbstractCandidateGenerator, create_func='create')
    factory.add_object_hierarchy("trainable_type_filter", _AbstractTrainableTypeFilter, create_func='create_for_training')
    factory.add_object_hierarchy("type_filter", _AbstractTypeFilter, create_func='create')
    factory.add_object_hierarchy("extractor", FeatureExtractorMixin, create_func='create')

    return factory