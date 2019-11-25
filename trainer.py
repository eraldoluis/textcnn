#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This example demonstrates the use of Conv1D for CNN text classification.
Original paper could be found at: https://arxiv.org/abs/1408.5882

This is the baseline model: CNN-rand.

The implementation is based on PyTorch.

We didn't implement cross validation,
but simply run `python cnn_mxnet.py` for multiple times,
the average accuracy is close to 78%.

It takes about 2 minutes for training 20 epochs on a GTX 970 GPU.
"""
import datetime
import json
import os
import time
from pprint import pprint

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Trainer(object):

    def __init__(self, config, model, train_corpus, valid_corpus, test_corpus, criterion, optimizer, use_cuda=False,
                 output_dir=None, verbose=True, train_metrics=None, val_metrics=None, test_metrics=None,
                 selection_metric=None):
        self.verbose = verbose
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.selection_metric = selection_metric

        self.use_cuda = use_cuda

        # FILES
        self.config = config
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

        # DATA
        self.train_data = TensorDataset(torch.LongTensor(train_corpus.x_data), torch.LongTensor(train_corpus.y_data))
        self.valid_data = TensorDataset(torch.LongTensor(valid_corpus.x_data), torch.LongTensor(valid_corpus.y_data))
        if test_corpus is not None:
            self.test_data = TensorDataset(torch.LongTensor(test_corpus.x_data), torch.LongTensor(test_corpus.y_data))

        # MODEL
        self.model = model

        # if self.verbose:
        #     print(self.corpus)
        #     print(self.model)

        self.num_epochs = config.num_epochs

        # Optimizer and Loss Function
        self.criterion = criterion
        self.optimizer = optimizer

    @staticmethod
    def get_time_dif(start_time):
        end_time = time.time()
        time_dif = end_time - start_time
        return datetime.timedelta(seconds=int(round(time_dif)))

    def evaluate(self, data, metrics, batch_size=64):
        """
        Evaluation, return accuracy and loss
        """
        if metrics is None:
            return {}

        y_true, y_pred = [], []

        self.model.eval()  # set mode to evaluation to disable dropout
        for data, label in DataLoader(data, batch_size=batch_size):
            # data, label = data.clone().detach(), label.clone().detach()
            if self.use_cuda:
                data, label = data.cuda(), label.cuda()

            with torch.no_grad():
                output = self.model(data)

            pred = torch.max(output, dim=1)[1]  # torch.max -> (value, index)

            y_pred.extend(pred.tolist())
            y_true.extend(label.tolist())

        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)

        res = {}
        for name, func in metrics.items():
            res[name] = func(y_true, y_pred)

        return res

    def train(self, tqdm_prefix=None, patience=0, init_res_dict=None):
        """
        Train and evaluate the model with training and validation data.
        """
        print("Training and evaluating...")

        tqdm_desc = []
        if tqdm_prefix:
            tqdm_desc.append(tqdm_prefix)
        tqdm_desc.append("Epoch {}/{}")

        res = {
            "best": {
                "selection_metric": 0.0,
                "epoch": 0,
                "valid": {},
                "train": {}
            },
            "perepoch": []
        }

        if init_res_dict is not None:
            res = dict(init_res_dict, **res)

        if self.use_cuda:
            self.model.cuda()

        if self.output_dir:
            res_file = open(os.path.join(self.output_dir, 'res_perepoch.json'), 'wt')
##barbara
        scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.0001, max_lr=0.001)
        iteration = 0
        for epoch in range(self.num_epochs):
            train_loader = DataLoader(self.train_data, batch_size=self.config.batch_size)

            self.model.train()
            for x_batch, y_batch in tqdm(train_loader, desc="\t".join(tqdm_desc).format(epoch + 1, self.num_epochs)):
                if self.use_cuda:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(x_batch)  # forward computation
                loss = self.criterion(outputs, y_batch)

                # backward propagation and update parameters
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                iteration += len(x_batch)

            # evaluate on both training and test dataset
            res_epoch = {"improved": False,
                         "epoch": epoch,
                         "iteration": iteration,
                         "train": self.evaluate(self.train_data, self.train_metrics),
                         "valid": self.evaluate(self.valid_data, self.val_metrics)}

            if init_res_dict is not None:
                res_epoch = dict(init_res_dict, **res_epoch)

            # store the best result
            if self.selection_metric and res_epoch["valid"][self.selection_metric] > res["best"]["selection_metric"]:
                res_epoch["improved"] = True
                res["best"]["selection_metric"] = res_epoch["valid"][self.selection_metric]
                res["best"]["epoch"] = epoch
                res["best"]["iteration"] = iteration
                res["best"]["train"] = res_epoch["train"]
                res["best"]["valid"] = res_epoch["valid"]
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.bin'))

            res["perepoch"].append(res_epoch)

            if self.verbose:
                # fmt_str = ["{}={}"] + ["{}={:f}"] * (len(res) - 1)
                # args = []
                # for name, val in res.items():
                #     args += [name, val]
                # print('\t'.join(fmt_str).format(*args), flush=True)

                print("*** Best Model according to %s ***" % self.selection_metric)
                pprint(res_epoch)

            if res_file:
                res_file.write(json.dumps(res_epoch) + '\n')
                res_file.flush()

            if patience > 0 and epoch - res["best"]["epoch"] > patience:
                if self.verbose:
                    print("Stopping on epoch {} due to no improvement since epoch {} (patience is {})".format(epoch,
                                                                                                              res[
                                                                                                                  "best"][
                                                                                                                  "epoch"],
                                                                                                              patience))
                break

        if res_file:
            res_file.close()

        if self.output_dir:
            # Write best model results.
            with open(os.path.join(self.output_dir, "res_best.json"), "wt") as f:
                json.dump(res["best"], f)

        return res

    # def test(self, test_data):
    #     """
    #     Test the model on test dataset.
    #     """
    #     if self.verbose:
    #         print(self.config.num_epochs)
    #         print(self.config.learning_rate)
    #
    #     print("Testing...")
    #     start_time = time.time()
    #     test_loader = DataLoader(test_data, batch_size=self.config.batch_size)
    #
    #     # load config and vocabulary
    #
    #     # restore the best parameters
    #     self.model.load_state_dict(torch.load(self.model_file, map_location=lambda storage, loc: storage))
    #
    #     y_true, y_pred = [], []
    #     for data, label in test_loader:
    #         data, label = torch.tensor(data), torch.tensor(label)
    #         if self.use_cuda:
    #             data, label = data.cuda(), label.cuda()
    #
    #         with torch.no_grad():
    #             output = self.model(data)
    #             pred = torch.max(output, dim=1)[1].cpu().numpy().tolist()
    #         y_pred.extend(pred)
    #         y_true.extend(label.cpu().numpy().tolist())
    #
    #     test_acc = metrics.accuracy_score(y_true, y_pred)
    #     test_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    #     if self.verbose:
    #         print("Test accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(test_acc, test_f1))
    #         # g_file.write("Test accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(test_acc, test_f1))
    #
    #         print("Precision, Recall and F1-Score...")
    #
    #     labels = np.array(range(len(self.config.target_names)))
    #     cm = metrics.confusion_matrix(y_true, y_pred)
    #
    #     if self.verbose:
    #         print(metrics.classification_report(y_true, y_pred, labels=labels, target_names=self.config.target_names))
    #         print('Confusion Matrix...')
    #         print(cm)
    #         print("Time usage:", self.get_time_dif(start_time))
    #
    #     return test_acc
