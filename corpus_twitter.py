"""
Corpus referente ao dataset de reconhecimento de entidades (Sao Paulo, Bahia, etc.)
"""

import numpy as np

from corpus import Corpus
from corpus_helper import CorpusHelper


class CorpusTE(Corpus):
    def __init__(self, train_file, vocab_file, sent_max_length=50):
        super().__init__(sent_max_length)
        self.train_file = train_file
        self.vocab_file = vocab_file
        self.label_to_id = {'Nao': 0, 'Sim': 1}
        self.max_labels = len(self.label_to_id)

        x_data = []
        y_data = []
        with CorpusHelper.open_file(train_file) as f:
            for line in f:
                line = line.strip()
                ftrs = line.split('\t')
                text = ftrs[1]
                label = ftrs[-1]
                x_data.append(text)
                ex_label = self.label_to_id[label.split()[0]]  # get only the first hashtag
                y_data.append(ex_label)

        self.words, self.word_id = CorpusHelper.read_vocab(vocab_file)
        self.vocab_size = len(self.word_id)

        for i in range(len(x_data)):
            # tokenization and padding
            x_data[i] = CorpusHelper.process_text(x_data[i], self.word_id, self.sentence_size, clean=False)

        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)

        self.size = len(y_data)

    def __str__(self):
        return '{}\n\tTraining: {}\n\tValidation: {}\n\tTesting: {}\n\tVocabulary: {}\n\tSentence: {}'.format(
            self.__class__,
            len(self.x_data),
            len(self.x_validation),
            len(self.y_test) if self.y_test else 0,
            self.vocab_size,
            self.sentence_size)
