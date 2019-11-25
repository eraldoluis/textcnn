import datetime
import os
from pprint import pprint

from torch import nn, optim

from corpus_twitter import CorpusTE
from textcnn import TextCNNConfig, TextCNN, ETextCNN
from trainer import Trainer
from utils import load_glove_embedding
import torch
from sklearn import metrics as skmetrics


def run(config, output_dir, num_splits=5, valid_split=0.2, patience=0):
    use_cuda = torch.cuda.is_available()
    vocab_file = 'data/twitter_hashtag/1kthashtag.vocab'
    dataset_file = 'data/DataSetsEraldo/dataSetSupernatural.txt'
    emb = load_glove_embedding('data/twitter_hashtag/1kthashtag.glove')

    criterion = nn.CrossEntropyLoss()

    corpus = CorpusTE(train_file=dataset_file, vocab_file=vocab_file)

    metrics = {'accuracy': skmetrics.accuracy_score, 'fscore_class1': skmetrics.f1_score}

    mean = 0.0
    for split in range(1, num_splits + 1):
        model = TextCNN(config=config, pre_trained_emb=emb)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        if config.stratified:
            train_corpus, valid_corpus = corpus.stratified_split(valid_split=valid_split)
        else:
            train_corpus, valid_corpus = corpus.split(valid_split=valid_split)

        output_dir_split = os.path.join(output_dir, "split{}".format(split))

        t = Trainer(train_corpus=train_corpus, valid_corpus=valid_corpus, test_corpus=None, model=model, config=config,
                    criterion=criterion, optimizer=optimizer, verbose=False, output_dir=output_dir_split,
                    train_metrics=metrics, val_metrics=metrics, selection_metric='fscore_class1', use_cuda=use_cuda)
        res = t.train(tqdm_prefix="Split {}/{}".format(split, num_splits), patience=patience,
                      init_res_dict={"split": split})
        pprint(res["best"])
        mean = mean + res['best']['selection_metric']
    mean = mean/num_splits
    print(mean)


if __name__ == '__main__':
    config = TextCNNConfig()
    config.num_epochs = 40
    config.batch_size = 128
    config.stratified = False
    config.balanced = True
    config.stratified_batch = False
    output_dir = "results/out_cv_shuffle_split_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    run(config, output_dir, num_splits=20, patience=5)
