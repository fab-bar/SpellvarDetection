import unittest

import spellvardetection.type_filter

class TestGeneratorFactory(unittest.TestCase):

    def test_factory_with_unknown_type(self):

        with self.assertRaises(ValueError):
            spellvardetection.type_filter.createTypeFilter({"type": "Unknown Type"})

    def test_train_factory_with_unknown_type(self):

        with self.assertRaises(ValueError):
            spellvardetection.type_filter.createTrainableTypeFilter({"type": "Unknown Type"})

    def test_factory_with_missing_options(self):

        with self.assertRaises(ValueError):
            spellvardetection.type_filter.createTypeFilter({"type": "sklearn"})

    def test_train_factory_with_missing_options(self):

        with self.assertRaises(ValueError):
            spellvardetection.type_filter.createTrainableTypeFilter({"type": "sklearn"})

    def test_train_factory_is_not_trained(self):

        filter_ = spellvardetection.type_filter.createTrainableTypeFilter({
            "type": "sklearn",
            "options": {"classifier_clsname": "__svm__", "feature_extractors": []}})
        with self.assertRaises(RuntimeError):
            filter_ = filter_.isPair("a", "b")

    def test_train_factory_for_sklearn_filter(self):

        self.assertIsInstance(
            spellvardetection.type_filter.createTrainableTypeFilter({
                "type": "sklearn",
                "options": {"classifier_clsname": "__svm__", "feature_extractors": []}}),
            spellvardetection.type_filter.SKLearnClassifierBasedTypeFilter
        )
