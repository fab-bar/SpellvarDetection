# -*- coding: utf-8 -*-

import csv

import numpy

class WordEmbeddings:

    def __init__(self, embeddings_type, embeddings_file, simplification_file=None, missing_words='zeros'):

        self.simplifications = {}

        if simplification_file is not None:
            with open(simplification_file, 'r', encoding='utf-8') as infile:
                csvreader = csv.reader(infile, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
                for row in csvreader:
                    self.simplifications[row[0]] = row[1]

        self.embeddings_type = embeddings_type
        if self.embeddings_type == 'hyperwords':
            self.embeddings = {}
            with open(embeddings_file, 'r', encoding='utf-8') as embeddfile:
                csvreader = csv.reader(embeddfile, delimiter=" ", quoting=csv.QUOTE_NONE, quotechar="")
                for row in csvreader:
                    self.embeddings[row[0]] = numpy.array(row[1:]).astype(numpy.float)

            if len(self.embeddings.values()) == 0:
                raise ValueError('Embedding file contains no embeddings.')

            self.dim = len(list(self.embeddings.values())[0])
            if missing_words == 'zeros':
                self.missing = numpy.zeros(self.dim)
            else:
                self.missing = missing_words
        else:
            raise ValueError('Embeddings of type "' + self.embeddings_type + '" are not supported.')


    def get(self, word):

        if word in self.simplifications:
            word = self.simplifications[word]

        if self.embeddings_type == "hyperwords":
            return self.embeddings.get(word, self.missing)

    def getDim(self):

        return self.dim
