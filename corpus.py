#!/usr/bin/python
# -*- coding: utf-8 -*-
import copy

import numpy as np
import pandas

from sklearn.model_selection import StratifiedShuffleSplit


class Corpus(object):

    def __init__(self, sent_max_length=40):
        self.sentence_size = sent_max_length
        self.word_id = None
        self.words = None

        self.x_data = None
        self.y_data = None

    @staticmethod
    def distribution(y):  # TODO
        dt = pandas.DataFrame(y.reshape(-1, 1), columns=['class'])
        print(dt)
        dist = dt.groupby('class').get_group(1)
        print(dist)
        nd = dist.to_numpy().sum()
        dist = (1 / len(y)) * nd
        return dist

    def split(self, valid_split=0.3, test_split=None, shuffle=True):
        """
        Split this corpus into two or three other corpus: train, validation and test.

        :param valid_split:
        :param test_split:
        :param shuffle:
        :return:
        """
        len_data = len(self.x_data)

        x_data = self.x_data
        y_data = self.y_data

        # shuffle
        if shuffle:
            indices = np.random.permutation(np.arange(len_data))
            x_data = x_data[indices]
            y_data = y_data[indices]

        splits = []

        # test split
        if test_split is not None:
            split = int(len_data * (1.0 - test_split))

            # create a new corpus for test split
            test = copy.copy(self)
            test.x_data = x_data[split:]
            test.y_test = y_data[split:]
            splits.insert(0, test)

            # remove test examples
            x_data = x_data[:split]
            y_data = y_data[:split]
            len_data = len(x_data)

        # validation split
        if valid_split is not None:
            split = int(len_data * (1.0 - valid_split))

            # create new corpus for validation split
            valid = copy.copy(self)
            valid.x_data = x_data[split:]
            valid.y_data = y_data[split:]
            splits.insert(0, valid)

            # remove test examples
            x_data = x_data[:split]
            y_data = y_data[:split]

        # create new corpus for train split
        train = copy.copy(self)
        train.x_data = x_data
        train.y_data = y_data
        splits.insert(0, train)

        return splits

    def stratified_split(self, valid_split=0.3, test_split=None):
        """
        Split this corpus into two or three other corpus: train, validation and test. Splits are stratified regarding
        label distribution.

        :param valid_split:
        :param test_split:
        :return:
        """
        len_data = len(self.x_data)

        x_data = self.x_data
        y_data = self.y_data

        splits = []

        # test split
        if test_split is not None:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split)

            # create a new corpus for test split
            test = copy.copy(self)
            train_idxs, test_idxs = next(sss.split(x_data, y_data))
            test.x_data = x_data[test_idxs]
            test.y_data = y_data[test_idxs]
            splits.insert(0, test)

            # remove test examples
            x_data = x_data[train_idxs]
            y_data = y_data[train_idxs]

        # validation split
        if valid_split is not None:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=valid_split)

            # create new corpus for validation split
            valid = copy.copy(self)
            train_idxs, test_idxs = next(sss.split(x_data, y_data))
            valid.x_data = x_data[test_idxs]
            valid.y_data = y_data[test_idxs]
            splits.insert(0, valid)

            # remove test examples
            x_data = x_data[train_idxs]
            y_data = y_data[train_idxs]

        # create new corpus for train split
        train = copy.copy(self)
        train.x_data = x_data
        train.y_data = y_data
        splits.insert(0, train)

        return splits

    def sub_sampling(self, size):
        # pega o num de exem da classe menos contemplada no cjt de treino, seleciona a mesma qtd
        # da outra classe e entao pega 2n exemplos do dev e test. um novo split de tamanho 2n é gerado
        n_mask = np.logical_not(np.array(self.y_data, dtype=bool))
        mask = np.array(self.y_data, dtype=bool)

        pos = self.x_data[mask].reshape(-1, self.sentence_size)  # [:size]
        neg = self.x_data[n_mask].reshape(-1, self.sentence_size)  # [: pos.shape[0]]

        index_p = np.random.permutation(np.arange(pos.shape[0]))[:size]
        index_n = np.random.permutation(np.arange(neg.shape[0]))[:size]
        pos = pos[index_p]
        neg = neg[index_n]

        pos_y = np.ones(pos.shape[0])
        neg_y = np.zeros(neg.shape[0])

        x = np.append(pos, neg, axis=0)
        y = np.append(pos_y, neg_y, axis=0)

        # Shuffle train
        indices = np.random.permutation(np.arange(2*size))
        self.x_data = x[indices]
        self.y_data = y[indices]
