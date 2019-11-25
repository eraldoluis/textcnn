#Corpus referente ao dataset do twitter criado pelo Henrico(ICMC)
import os
import numpy as np

from corpus import Corpus
from corpus_helper import CorpusHelper


class CorpusTH(Corpus):

    def __init__(self, train_file, vocab_file, dev_split=0.2, test_split=None, sent_max_length=50, vocab_size=8000):
        super().__init__()
        self.train_file = train_file
        self.vocab_file = vocab_file

        self.dev_split = dev_split
        self.test_split = test_split
        self.label_to_id = {0: 0, 1: 1, 2: 2}
        self.max_labels = len(self.label_to_id)

    def prepare_data(self, args=None):
        x_data = []
        with CorpusHelper.open_file(self.train_file[0]) as f:
            for l in f:
                l = l.strip()
                x_data.append(l)
        f.close()
        len_neg = len(x_data)

        with CorpusHelper.open_file(self.train_file[1]) as f:
            for l in f:
                l = l.strip()
                x_data.append(l)
        f.close()
        len_pos = len(x_data)-len_neg

        with CorpusHelper.open_file(self.train_file[2]) as f:
            for l in f:
                l = l.strip()
                x_data.append(l)

        f.close()
        len_neu = len(x_data) - (len_pos + len_neg)

        self.words, self.word_id = CorpusHelper.read_vocab(self.vocab_file)
        self.vocab_size = len(self.word_id)
        for i in range(len(x_data)):  # tokenizing and padding
            x_data[i] = CorpusHelper.process_text(x_data[i], self.word_id,
                                                  self.sentence_size, clean=False)

        # print(x_data)
        x = np.array(x_data)
        pos = np.ones(len_pos)
        neg = np.zeros(len_neg)
        neu = 2*np.ones(len_neu)
        y = np.append(pos, neg)
        y = np.append(y, neu)
        y_data = np.array(y)
        size = len(y_data)

        # Shuffle train
        indices = np.random.permutation(np.arange(size))
        self.x_data = x[indices]
        self.y_data = y[indices]
