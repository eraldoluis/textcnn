"""
Adapted from: https://github.com/galatolofederico/pytorch-balanced-batch
"""
import random

import torch.utils.data
import pandas as pd

from sklearn.model_selection import StratifiedKFold


class BalancedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, labels):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(len(labels)):
            label = labels[idx]
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        return self.labels[idx].item()

    def __len__(self):
        return self.balanced_max * len(self.keys)


class BalancedSampler2(torch.utils.data.sampler.Sampler):
    def __init__(self, labels):
        self.labels = pd.Series(labels)
        self.unq_labels = self.labels.unique()
        self.num_unq_labels = len(self.unq_labels)
        self.unq_max = self.unq_labels.value_counts().max()

    def __iter__(self):
        pass

    def __len__(self):
        return self.unq_max * self.num_unq_labels


class StratifiedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size

        self.skf = StratifiedKFold(n_splits=(len(labels) // batch_size), shuffle=True)

    def __iter__(self):
        for _, test in self.skf.split(self.labels, self.labels):
            yield test

    def __len__(self):
        return self.skf.n_splits
