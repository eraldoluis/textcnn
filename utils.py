import torch


def load_glove_embedding(glove_file, size=100):
    with open(glove_file, 'r') as file:
        file.readline()
        data = []
        for line in file:
            data.append([float(val) for val in line.split()])

    return torch.tensor(data)
