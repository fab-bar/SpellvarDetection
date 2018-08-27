# -*- coding: utf-8 -*-

import csv

class WordClusters:

    def __init__(self, cluster_type, cluster_file, unknown_type=None):

        self.cluster_words = {}
        self.words_cluster = {}

        self.cluster_type = cluster_type
        self.unknown_type = unknown_type

        if self.cluster_type == 'brown':
            with open(cluster_file, 'r', encoding='utf-8') as infile:
                csvreader = csv.reader(infile, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
                for row in csvreader:
                    self.words_cluster[row[1]] = row[0]
                    self.cluster_words.setdefault(row[0] ,set()).add(row[1])

        else:
            raise ValueError('Clusters of type "' + self.cluster_type + '" are not supported.')

        if self.unknown_type is not None and self.unknown_type not in self.words_cluster:
            raise ValueError('Unknown type ("' + self.unknown_type + '") must be in vocabulary.')

    def isOOV(self, word):

        return word not in self.words_cluster

    def hasCluster(self, word):

        return self.getCluster(word) is not None

    def getCluster(self, word):

        if not self.isOOV(word):
            return self.words_cluster[word]

        elif self.unknown_type is not None:
            return self.words_cluster[self.unknown_type]

        else:
            return None

    def inSameCluster(self, word_a, word_b):

        return self.getCluster(word_a) == self.getCluster(word_b)
