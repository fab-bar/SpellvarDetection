import unittest

from spellvardetection.util.factory import Factory

class TestGeneratorFactory(unittest.TestCase):

    def test_create_object_unkown_name(self):

        with self.assertRaises(ValueError):
            Factory().create_from_name("unknown_name", None)

    def test_create_object_unkown_class(self):

        with self.assertRaises(ValueError):
            Factory().create_from_cls(str, None)

    def test_register_name_twice(self):

        factory = Factory()
        factory.add_object_hierarchy('string', str)
        with self.assertRaises(ValueError):
            factory.add_object_hierarchy('string', str)

    def test_register_class_twice(self):

        factory = Factory()
        factory.add_object_hierarchy('string', str)
        with self.assertRaises(ValueError):
            factory.add_object_hierarchy('another string', str)

    def test_register_class_factory_twice(self):

        factory = Factory()
        factory.add_factory_method(str, None)
        with self.assertRaises(ValueError):
            factory.add_factory_method(str, None)
