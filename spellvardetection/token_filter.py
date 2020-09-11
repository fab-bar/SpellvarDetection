import abc
import csv
import multiprocessing
import os
import functools
import random
import pickle

import numpy
import numpy.random

import tensorflow

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, concatenate, Reshape

import joblib

import spellvardetection.lib.util
import spellvardetection.lib.embeddings

class _AbstractTokenFilter(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def isPair(self, word, candidate, left_context, right_context):  # pragma: no cover
        pass

    def filterCandidates(self, word, candidates, left_context, right_context):

        return set([candidate for candidate in candidates if self.isPair(word, candidate, left_context, right_context)])


class _AbstractTrainableTokenFilter(_AbstractTokenFilter):

    def _getClassifier(self):

        try: self.clf
        except AttributeError: raise Exception("Train or load a model first.")

        return self.clf

    @abc.abstractmethod
    def train(self, positive_pairs, negative_pairs):  # pragma: no cover
        pass

    @abc.abstractmethod
    def load(modelfile_name):  # pragma: no cover
        pass

    @abc.abstractmethod
    def save(self, modelfile_name):  # pragma: no cover
        pass

class CNNTokenFilter(_AbstractTrainableTokenFilter):

    name = 'cnn'

    def create(modelfile_name: os.PathLike):
        return CNNTokenFilter.load(modelfile_name)


    def __init__(self, left_context_len, right_context_len,
                 use_form_embedding=True, vector_type=None, vectorfile_name: os.PathLike = None,  simplfile_name: os.PathLike = None,
                 filter_lengths: list = None, char_filter_lengths: list = None, nb_filter=50, max_word_len=12, char_embedding_dim=10,
                 batch_size=20, epochs=10, seed=None
    ):

        if seed is not None:
            numpy.random.seed(seed)
            random.seed(seed)
            tensorflow.random.set_seed(seed)

        self.left_context_len = left_context_len
        self.right_context_len = right_context_len

        self.use_form_embedding = use_form_embedding

        if vector_type is not None:
            self.embeddings = spellvardetection.lib.embeddings.WordEmbeddings(vector_type, vectorfile_name, simplfile_name)
            self.use_context_embedding = True
        else:
            self.use_context_embedding = False

        ## the longest filter should span the whole context to one side + the target word
        if filter_lengths is None:
            self.filter_lengths = range(2,max(self.left_context_len, self.right_context_len) + 2)
        else:
            self.filter_lengths = filter_lengths
        if char_filter_lengths is None:
            self.char_filter_lengths = [2,3]
        else:
            self.char_filter_lengths = char_filter_lengths

        self.max_word_len = max_word_len
        self.char_embedding_dim = char_embedding_dim
        self.nb_filter = nb_filter

        self.batch_size = batch_size
        self.epochs = epochs

        if not self.use_context_embedding and not self.use_form_embedding:
            raise ValueError("Needs to use form embeddings and/or pretrained word embeddings.")


    def _getSequence(self, word, left_context, right_context):

        left_sequence = [self.embeddings.get(cword) for cword in left_context[-self.left_context_len:]]
        if len(left_sequence) < self.left_context_len:
            left_sequence = [numpy.zeros(self.embeddings.getDim())]*(self.left_context_len - len(left_sequence)) + left_sequence

        right_sequence = [self.embeddings.get(cword) for cword in right_context[:self.right_context_len]]
        if len(right_sequence) < self.right_context_len:
            right_sequence = right_sequence + [numpy.zeros(self.embeddings.getDim())]*(self.right_context_len - len(right_sequence))

        return numpy.concatenate((
            left_sequence,
            [self.embeddings.get(word)],
            right_sequence), axis=0)


    def _getInput(self, candidate, left_context, right_context):

        X = []

        if self.use_context_embedding:
            X .append(numpy.array([self._getSequence(candidate, left_context, right_context)]))

        if self.use_form_embedding:

            left_context = ['']*(self.left_context_len-len(left_context)) + left_context[-self.left_context_len:]
            right_context = right_context[:self.right_context_len] + ['']*(self.right_context_len-len(right_context))
            word_sequences = [pad_sequences([seq], maxlen=self.max_word_len, padding='post', truncating='post')
                              for seq in self.char_tokenizer.texts_to_sequences(left_context + [candidate] + right_context)]

            X += word_sequences

        return X


    def _buildNetwork(self, nb_filter, filter_lengths, char_filter_lengths):

        input_list = []
        sequence = []

        if self.use_context_embedding:
            c_embedd_in = Input(shape=(self.left_context_len + self.right_context_len + 1, self.embeddings.getDim()), dtype='float')
            input_list.append(c_embedd_in)
            sequence.append(c_embedd_in)


        if self.use_form_embedding:
            char_inputs = []
            char_word_embeddings = []
            char_embedd = Embedding(len(self.char_tokenizer.word_index) + 1, self.char_embedding_dim)
            filters = [Conv1D(nb_filter, length, padding='same', activation="relu") for length in char_filter_lengths]
            pooling = GlobalMaxPooling1D()
            char_reshape = Reshape((1,-1))

            for i in range(self.left_context_len + self.right_context_len + 1):
                char_input = Input(shape=(self.max_word_len,), dtype='int32')
                char_inputs.append(char_input)

                conv_features = []
                for filter in filters:
                    features = pooling(filter(char_embedd(char_input)))

                    conv_features.append(features)

                e = concatenate(conv_features)
                char_word_embeddings.append(char_reshape(e))

            char_embedding_sequence = concatenate(char_word_embeddings, axis=1)
            sequence.append(char_embedding_sequence)
            input_list += char_inputs

        if len(sequence) > 1:
            sequence = concatenate(sequence)
        else:
            sequence = sequence[0]

        conv_features = []
        for length in filter_lengths:

            filter = Conv1D(nb_filter, length, padding='same', activation="relu")(sequence)
            features = GlobalMaxPooling1D()(filter)

            conv_features.append(features)

        if len(conv_features) > 1:
            features = concatenate(conv_features)
        else:
            features = conv_features[0]

        predictions = Dense(1, activation='sigmoid')(features)

        if len(input_list) > 1:
            return Model(inputs=input_list, outputs=predictions)
        else:
            return Model(inputs=input_list[0], outputs=predictions)


    def train(self, positive_pairs, negative_pairs):

        if self.use_form_embedding:
            self.char_tokenizer = Tokenizer(num_words=None, filters='', lower=True, split="", char_level=True)
            self.char_tokenizer.fit_on_texts(
                ["".join([pair[1]] + pair[2] + pair[3]) for pair in positive_pairs + negative_pairs]
            )

        self.clf = self._buildNetwork(self.nb_filter, self.filter_lengths, self.char_filter_lengths)

        self.clf.compile(optimizer='adam',
                         loss='binary_crossentropy')


        def batch_generator(positive_pairs, negative_pairs, positive_batch_size):
            while True:
                batch = (random.sample(positive_pairs, k=positive_batch_size) + random.sample(negative_pairs, k=positive_batch_size))
                input_list = [self._getInput(pair[1], pair[2], pair[3]) for pair in batch]
                X = [numpy.concatenate(inp) for inp in zip(*input_list)]

                Y = numpy.array([1]*positive_batch_size + [0]*positive_batch_size)
                yield (X, Y)

        gen = batch_generator(positive_pairs, negative_pairs, min(self.batch_size, len(positive_pairs)))
        self.clf.fit(
            x=batch_generator(positive_pairs, negative_pairs, min(self.batch_size, len(positive_pairs))),
            steps_per_epoch=len(positive_pairs)/self.batch_size, epochs=self.epochs)

    def __getstate__(self):

        state = self.__dict__.copy()
        del state['clf']
        return state

    def save(self, modelfile_name):
        joblib.dump(self, modelfile_name)
        self.clf.save(modelfile_name + '.tf')

    def load(modelfile_name):
        obj = joblib.load(modelfile_name)
        obj.clf = tensorflow.keras.models.load_model(modelfile_name + '.tf')
        return obj


    def isPair(self, word, candidate, left_context, right_context):

        input_sequence = self._getInput(candidate, left_context[-self.left_context_len:], right_context[:self.right_context_len])
        if len(input_sequence) == 1:
            input_sequence = input_sequence[0]

        return self._getClassifier().predict(input_sequence)[0][0] > 0.5
