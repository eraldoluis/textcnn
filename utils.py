import csv

import torch
import numpy as np


def load_glove_embedding(glove_file, size=100):
    with open(glove_file, 'r') as g_file:
        reader = csv.reader(g_file, delimiter=' ')
        reader.__next__()
        data = []
        for l in reader:
            data.append([l])

    data = np.asarray(data, dtype=float)
    return torch.FloatTensor(data.reshape((-1, size)))
