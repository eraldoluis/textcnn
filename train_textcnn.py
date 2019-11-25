import datetime
from pprint import pprint

from torch import nn, optim

from corpus_twitter import CorpusTE
from textcnn import TextCNNConfig, TextCNN,  ETextCNN
from trainer import Trainer
from utils import load_glove_embedding

from sklearn import metrics as skmetrics


def train():
    vocab_file = 'data/twitter_hashtag/1kthashtag.vocab'
    dataset_file = 'data/DataSetsEraldo/dataSetSupernatural.txt'

    config = TextCNNConfig()

    config.num_epochs = 50
    config.batch_size = 128
    config.stratified = True

    emb = load_glove_embedding('data/twitter_hashtag/1kthashtag.glove')

    model = ETextCNN(config=config, pre_trained_emb=emb)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    output_dir = "results/out_train_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))

    metrics = {'accuracy': skmetrics.accuracy_score, 'fscore_class1': skmetrics.f1_score}

    corpus = CorpusTE(train_file=dataset_file, vocab_file=vocab_file)
    if config.stratified:
        train_corpus, valid_corpus = corpus.stratified_split(valid_split=0.2)
    else:
        train_corpus, valid_corpus = corpus.split(valid_split=0.2)

    t = Trainer(train_corpus=train_corpus, valid_corpus=valid_corpus, test_corpus=None, model=model, config=config,
                criterion=criterion, optimizer=optimizer, verbose=True, output_dir=output_dir, train_metrics=metrics,
                val_metrics=metrics, selection_metric='fscore_class1')
    res = t.train(patience=5)
    pprint(res["best"])


if __name__ == '__main__':
    train()
