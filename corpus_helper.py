#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
import re
import os
import csv
import pandas


class CorpusHelper(object):
    @staticmethod
    def open_file(filename, mode='r'):
        """
        Commonly used file reader and writer, change this to switch between python2 and python3.
        :param filename: filename
        :param mode: 'r' and 'w' for read and write respectively
        """
        return open(filename, mode, encoding='utf-8', errors='ignore')

    @staticmethod
    def clean(text):
        text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " \( ", text)
        text = re.sub(r"\)", " \) ", text)
        text = re.sub(r"\?", " \? ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip().lower()

    @staticmethod
    def build_vocab(data, vocab_file, vocab_size=0):
        """
           Build vocabulary file from training data.
           """
        print('Building vocabulary...')
        all_data = []  # group all data
        for content in data:
            all_data.extend(content.split())

        counter = Counter(all_data)  # count and get the most common words
        count_pairs = counter.most_common(vocab_size - 1)
        words, _ = list(zip(*count_pairs))

        words = ['<PAD>'] + list(words)  # add a padding with id 0 to pad the sentence to same length
        CorpusHelper.open_file(vocab_file, 'w').write('\n'.join(words) + '\n')

    @staticmethod
    def read_vocab(vocab_file):
        """
        Read vocabulary from file.
        One word per line.
        """
        words = CorpusHelper.open_file(vocab_file).read().strip().split('\n')
        word_to_id = dict(zip(words, range(len(words))))
        return words, word_to_id

    @staticmethod
    def process_text(text, word_to_id, max_length, clean=True):
        """tokenizing and padding"""
        if clean:  # if the data needs to be cleaned
            text = CorpusHelper.clean(text)
        text = text.split()

        text = [word_to_id[x] for x in text if x in word_to_id]
        if len(text) < max_length:
            text = [0] * (max_length - len(text)) + text
        return text[:max_length]
