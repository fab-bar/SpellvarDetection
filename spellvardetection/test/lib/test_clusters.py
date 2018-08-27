import unittest
from unittest import mock

from spellvardetection.lib.clusters import WordClusters

def my_open_mock(filename, rw, encoding=None):
    if filename == 'empty_file':
        content = ""
    elif filename == 'brown_file':
        content = '0	the	6\n10	chased	3\n110	dog	2\n1110	mouse	2\n110	cat	2\n110	<UNK>	5'
    else:
        raise FileNotFoundError(filename)
    file_object = mock.mock_open(read_data=content).return_value
    file_object.__iter__.return_value = content.splitlines(True)
    return file_object

@mock.patch('spellvardetection.lib.clusters.open', new=my_open_mock)
class TestWordEmbeddings(unittest.TestCase):

    def test_cluster_type_not_supported(self):

        with self.assertRaises(ValueError):
            WordClusters('Unknown type', 'brown_file')

    def test_clusters_in_same_cluster(self):

        self.assertTrue(WordClusters('brown', 'brown_file').inSameCluster('cat', 'dog'))

    def test_clusters_not_in_same_cluster(self):

        self.assertFalse(WordClusters('brown', 'brown_file').inSameCluster('cat', 'mouse'))

    def test_clusters_unknown_not_in_vocabulary(self):

        with self.assertRaises(ValueError):
            WordClusters('brown', 'brown_file', unknown_type="unknown type")

    def test_clusters_unknown_in_same_cluster(self):

        self.assertTrue(WordClusters('brown', 'brown_file', unknown_type="<UNK>").inSameCluster('cat', 'unknown type'))

    def test_clusters_unknown_not_in_same_cluster(self):

        self.assertFalse(WordClusters('brown', 'brown_file', unknown_type="<UNK>").inSameCluster('mouse', 'unknown type'))

    def test_clusters_unknown_without_unknown_type(self):

        self.assertTrue(WordClusters('brown', 'brown_file').inSameCluster('unknown type', 'another unknown'))

    def test_is_oov(self):

        self.assertTrue(WordClusters('brown', 'brown_file', unknown_type="<UNK>").isOOV('unknown type'))

    def test_has_cluster_known(self):

        self.assertTrue(WordClusters('brown', 'brown_file', unknown_type="<UNK>").hasCluster('cat'))

    def test_has_cluster_unknown_with_unknown_type(self):

        self.assertTrue(WordClusters('brown', 'brown_file', unknown_type="<UNK>").hasCluster('unknown type'))

    def test_has_cluster_unknown_without_unknown_type(self):

        self.assertFalse(WordClusters('brown', 'brown_file').hasCluster('unknown type'))
