#!/usr/bin/python
# -*- coding: utf-8 -*-
import copy

import numpy as np
import pandas


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
