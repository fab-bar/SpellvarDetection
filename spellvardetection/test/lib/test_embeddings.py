import unittest
from unittest import mock

from spellvardetection.lib.embeddings import WordEmbeddings

def my_open_mock(filename, rw, encoding=None):
    if filename == 'empty_file':
        content = ""
    elif filename == 'embedd_file':
        content = 'a 1 2'
    else:
        raise FileNotFoundError(filename)
    file_object = mock.mock_open(read_data=content).return_value
    file_object.__iter__.return_value = content.splitlines(True)
    return file_object

@mock.patch('spellvardetection.lib.embeddings.open', new=my_open_mock)
class TestWordEmbeddings(unittest.TestCase):

    def test_embeddings_type_not_supported(self):

        with self.assertRaises(ValueError):
            WordEmbeddings('Unknown type', 'embedd_file')

    def test_hyperword_embeddings_file_empty(self):

        with self.assertRaises(ValueError):
            WordEmbeddings('hyperwords', 'empty_file')

    def test_hyperword_embeddings_dimension(self):

        self.assertEquals(2, WordEmbeddings('hyperwords', 'embedd_file').getDim())

    def test_hyperword_embeddings_get_known_embedding(self):

        self.assertEquals([1.,2.], WordEmbeddings('hyperwords', 'embedd_file').get('a').tolist())

    def test_hyperword_embeddings_get_unknown_embedding(self):

        self.assertEquals([0.,0.], WordEmbeddings('hyperwords', 'embedd_file').get('z').tolist())
        self.assertIsNone(WordEmbeddings('hyperwords', 'embedd_file', missing_words=None).get('z'))

