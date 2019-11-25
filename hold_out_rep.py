import datetime
import os
import torch
from pprint import pprint
from torch import nn, optim
from corpus import Corpus
from corpus_twitter_hashtags import TwitterHashtagCorpus
from corpus_twitter import CorpusTE
from textcnn import TextCNNConfig, TextCNN, ETextCNN
from trainer import Trainer
from utils import load_glove_embedding

from sklearn import metrics as skmetrics


def run(config, output_dir, num_rep=5, valid_split=0.2, patience=0):
    use_cuda = torch.cuda.is_available()

    mean = 0.0 ## barbara
    vocab_file = 'data/twitter_hashtag/1kthashtag.vocab'
    dataset_file = 'data/twitter_hashtag/multiple.txt'
    emb = load_glove_embedding('data/twitter_hashtag/1kthashtag.glove')

    criterion = nn.CrossEntropyLoss()

    corpus = TwitterHashtagCorpus(train_file=dataset_file, vocab_file=vocab_file)
    config.vocab_size = corpus.vocab_size
    train_corpus = Corpus()
    train_corpus.x_data = corpus.x_train[:1000]
    train_corpus.y_data = corpus.y_train[:1000]
    valid_corpus = Corpus()
    valid_corpus.x_data = corpus.x_validation[:1000]
    valid_corpus.y_data = corpus.y_validation[:1000]

    metrics = {'accuracy': skmetrics.accuracy_score}

    for rep in range(1, num_rep + 1):
        model = TextCNN(config=config, pre_trained_emb=emb)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        #train_corpus, valid_corpus = corpus.split(valid_split=valid_split)

        output_dir_rep = os.path.join(output_dir, "rep{}".format(rep))

        t = Trainer(train_corpus=train_corpus, valid_corpus=valid_corpus, test_corpus=None, model=model, config=config,
                    criterion=criterion, optimizer=optimizer, verbose=False, output_dir=output_dir_rep,
                    train_metrics=metrics, val_metrics=metrics, selection_metric='accuracy', use_cuda=use_cuda)
        res = t.train(tqdm_prefix="Rep {}/{}".format(rep, num_rep), patience=patience,
                      init_res_dict={"rep": rep})

        pprint(res["best"])
        mean = mean + res['best']['selection_metric']
    mean = mean/num_rep
    print(mean)


if __name__ == '__main__':
    for k in range(2, 11):
        config = TextCNNConfig()
        config.num_epochs = 40
        config.batch_size = 256
        config.kernel_sizes = [k]
        config.num_classes = 125
        output_dir = "../../experiments/results/out_rp_kernels_1kt_{}{}".format(k, datetime.datetime.now().strftime( "%Y%m%d-%H%M%S-%f"))
        run(config, output_dir, num_rep=10, patience=5)
