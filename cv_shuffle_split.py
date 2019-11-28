import datetime
import os
from pprint import pprint

import torch
from sklearn import metrics as skmetrics
from torch import nn, optim

from corpus_twitter import CorpusTE
from textcnn import TextCNNConfig, TextCNN
from trainer import Trainer
from utils import load_glove_embedding


def run(config, output_dir, num_splits=5, patience=0):
    use_cuda = torch.cuda.is_available() and config.cuda_device >= 0
    vocab_file = 'data/twitter_hashtag/1kthashtag.vocab'
    dataset_file = 'data/DataSetsEraldo/dataSetSupernatural.txt'

    # The returned embedding tensor is kept unchanged to init each split model.
    emb_ = load_glove_embedding('data/twitter_hashtag/1kthashtag.glove')

    criterion = nn.CrossEntropyLoss()

    corpus = CorpusTE(train_file=dataset_file, vocab_file=vocab_file)

    metrics = {'accuracy': skmetrics.accuracy_score, 'fscore_class1': skmetrics.f1_score}

    if config.stratified:
        def fun_split(vs):
            return corpus.stratified_split(vs)
    else:
        def fun_split(vs):
            return corpus.split(vs)

    mean = 0.0
    for split in range(1, num_splits + 1):
        # Create a copy of the embedding tensor to avoid information leak between splits.
        # It is important to call detach(), since clone() is recorded in the computation graph
        #   (gradients propagated to the cloned tensor will be propagated to the original tensor).
        emb = emb_.clone().detach()

        model = TextCNN(config=config, pre_trained_emb=emb)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        train_corpus, valid_corpus = fun_split(config.valid_split)

        output_dir_split = os.path.join(output_dir, "split{}".format(split))

        t = Trainer(train_corpus=train_corpus, valid_corpus=valid_corpus, test_corpus=None, model=model, config=config,
                    criterion=criterion, optimizer=optimizer, verbose=False, output_dir=output_dir_split,
                    train_metrics=metrics, val_metrics=metrics, selection_metric='fscore_class1', use_cuda=use_cuda)
        res = t.train(tqdm_prefix="Split {}/{}".format(split, num_splits), patience=patience,
                      init_res_dict={"split": split})
        pprint(res["best"])
        mean = mean + res['best']['selection_metric']
    mean = mean / num_splits
    print(mean)


if __name__ == '__main__':
    conf = TextCNNConfig()
    conf.learning_rate = 1e-2
    conf.num_epochs = 40
    conf.batch_size = 128
    conf.stratified = False
    conf.balanced = True
    conf.stratified_batch = False
    output_dir = "results/out_cv_shuffle_split_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    run(conf, output_dir, num_splits=5, patience=5)
