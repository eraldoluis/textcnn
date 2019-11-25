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


def run(config, output_dir, num_splits=5, valid_split=0.2, patience=0):
    use_cuda = torch.cuda.is_available()
    mean = 0.0 ## barbara
    vocab_file = 'data/twitter_hashtag/1kthashtag.vocab'
    dataset_file = 'data/DataSetsEraldo/dataSetSupernatural.txt'
    emb = load_glove_embedding('data/twitter_hashtag/1kthashtag.glove')

    criterion = nn.CrossEntropyLoss()

    corpus = CorpusTE(train_file=dataset_file, vocab_file=vocab_file)
    config.vocab_size = corpus.vocab_size
    metrics = {'accuracy': skmetrics.accuracy_score, 'fscore_class1': skmetrics.f1_score}

    for split in range(1, num_splits + 1):
        model = ETextCNN(config=config, pre_trained_emb=emb)
        model.encoder.use_pre_trained_layers('../results/out_rp_1kt_20191125-130924-229380/rep1/best_model.bin', False)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        train_corpus, valid_corpus = corpus.split(valid_split=valid_split)
        #train_corpus.sub_sampling(350)
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
    config.batch_size = 256
    config.kernel_sizes = [3, 4, 5]
    config.num_classes = 125
    output_dir = "../results/out_rp_1kt_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    run(config, output_dir, num_splits=5, patience=5)
