import datetime
from math import ceil
from pprint import pprint

from sklearn import metrics as skmetrics
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR

from corpus_twitter import CorpusTE
from textcnn import TextCNNConfig, TextCNN
from trainer import Trainer
from utils import load_glove_embedding


def train():
    vocab_file = 'data/twitter_hashtag/1kthashtag.vocab'
    dataset_file = 'data/DataSetsEraldo/dataSetSupernatural.txt'

    config = TextCNNConfig()

    config.batch_size = 128
    config.stratified = False
    config.balanced = True
    config.stratified_batch = False

    corpus = CorpusTE(train_file=dataset_file, vocab_file=vocab_file)
    if config.stratified:
        train_corpus, valid_corpus = corpus.stratified_split(valid_split=config.valid_split)
    else:
        train_corpus, valid_corpus = corpus.split(valid_split=config.valid_split)

    num_epochs = 12
    num_iter = num_epochs * ceil(len(train_corpus.y_data) / config.batch_size)
    lr_min = 1e-5
    lr_max = 1

    config.learning_rate = lr_min
    config.num_epochs = num_epochs

    emb = load_glove_embedding('data/twitter_hashtag/1kthashtag.glove')
    model = TextCNN(config=config, pre_trained_emb=emb)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    output_dir = "results/out_train_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))

    metrics = {'accuracy': skmetrics.accuracy_score, 'fscore_class1': skmetrics.f1_score}

    lr_scheduler = LambdaLR(optimizer, lambda it: (lr_max / lr_min) ** (it / (num_iter - 1)))

    t = Trainer(train_corpus=train_corpus, valid_corpus=valid_corpus, test_corpus=None, model=model, config=config,
                criterion=criterion, optimizer=optimizer, verbose=True, output_dir=output_dir, train_metrics=metrics,
                val_metrics=metrics, selection_metric='fscore_class1', lr_scheduler=lr_scheduler)
    res = t.train(patience=0)
    pprint(res["best"])


if __name__ == '__main__':
    train()
