import unittest

from spellvardetection.util.spellvarfactory import create_base_factory
import spellvardetection.type_filter

class TestGeneratorFactory(unittest.TestCase):

    factory = create_base_factory()

    def test_factory_with_unknown_type(self):

        with self.assertRaises(ValueError):
            self.factory.create_from_name("type_filter", {"type": "Unknown Type"})

    def test_train_factory_with_unknown_type(self):

        with self.assertRaises(ValueError):
            self.factory.create_from_name("trainable_type_filter", {"type": "Unknown Type"})

    def test_factory_with_missing_options(self):

        with self.assertRaises(ValueError):
            self.factory.create_from_name("type_filter", {"type": "sklearn"})

    def test_train_factory_with_missing_options(self):

        with self.assertRaises(ValueError):
            self.factory.create_from_name("trainable_type_filter", {"type": "sklearn"})

    def test_train_factory_is_not_trained(self):

        filter_ = self.factory.create_from_name("trainable_type_filter", {
            "type": "sklearn",
            "options": {"classifier_clsname": "__svm__", "feature_extractors": []}})
        with self.assertRaises(RuntimeError):
            filter_ = filter_.isPair("a", "b")

    def test_train_factory_for_sklearn_filter(self):

        self.assertIsInstance(
            self.factory.create_from_name("trainable_type_filter", {
                "type": "sklearn",
                "options": {"classifier_clsname": "__svm__", "feature_extractors": []}}),
            spellvardetection.type_filter.SKLearnClassifierBasedTypeFilter
        )
