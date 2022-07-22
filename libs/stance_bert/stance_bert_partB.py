"""
BERT model for stance detecting
"""

import os
import torch
import logging
import pandas as pd
import sys
import time

sys.path.append("../../")
sys.path.append("../")
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import BertTokenizer, AdamW, BertForSequenceClassification, AlbertTokenizer, AlbertModel
from libs.stance_bert.Bert_partB import StanceBerModel
from utils.data_during_training import TrainingDataCollector, Logger
from libs.stance_bert.model_train_config import Config

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_name = Config['data_name']


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path).fillna("")
        self.article = list(df['article'])
        self.title = list(df['title'])
        self.target = list(df['target'])
        self.label = list(df['label'])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.article[idx] + self.title[idx], self.target[idx], self.label[idx]


def get_dataloader(data_path, batch_size=Config["batch_size"]):
    dataset = Dataset(data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader


class Classifier:

    def __init__(self, train):
        """
        :param train: train or test
        """
        self.epoch_number = Config["epoch_number"]
        self.device = torch.device(f"cuda:{Config['cuda_index']}" if torch.cuda.is_available() else "cpu")
        self.plot = TrainingDataCollector()
        if Config['use_albert']:
            self.tokenizer = AlbertTokenizer.from_pretrained(Config["Tokenizer"])
        else:
            self.tokenizer = BertTokenizer.from_pretrained(Config["Tokenizer"])
        self.CELoss = torch.nn.CrossEntropyLoss()
        if train:
            self.model = StanceBerModel()
        else:
            self.model = BertForSequenceClassification.from_pretrained(path + "/model/")
            print("using fine-tune model")
        self.model.to(self.device)

    def train(self, train_loader, val_loader, lr=Config["lr"]):
        optimizer = AdamW(self.model.parameters(), lr)
        epoch_list = []
        epoch_acc_train, epoch_loss_train, epoch_acc_test, epoch_loss_test = [], [], [], []
        for epoch in range(self.epoch_number):
            batch_loss = 0
            epoch_list.append(epoch + 1)
            pred_list = []
            y_list = []
            self.model.train()
            if not os.path.exists(os.path.join(path, f"model/{data_name}")):
                os.makedirs(os.path.join(path, f"model/{data_name}"))

            for batch, (body, target, label) in enumerate(train_loader):
                x = self.tokenizer(list(zip(target, body)), return_tensors='pt', padding='max_length',
                                   truncation=True, max_length=Config["max_length"]).to(self.device)

                y = label.long().to(self.device)
                optimizer.zero_grad()
                output = self.model.forward(x)
                loss = self.CELoss(output, y)
                loss.backward()
                optimizer.step()
                y_pred = output.argmax(dim=1)
                batch_loss += loss.item()
                pred_list += y_pred.tolist()
                y_list += y.tolist()
                if (batch + 1) % 200 == 0:
                    pred_list = pred_list[:len(y_list)]
                    f1 = f1_score(y_true=y_list, y_pred=pred_list, average='macro')
                    pre = precision_score(y_true=y_list, y_pred=pred_list, average='macro')
                    recall = recall_score(y_true=y_list, y_pred=pred_list, average='macro')
                    print('[Epoch{} Batch{}] loss:{:.3f} precision:{:.3f} recall:{:.3f} f1:{:.3f}'.format(epoch + 1,
                                                                                                          batch + 1,
                                                                                                          batch_loss,
                                                                                                          pre,
                                                                                                          recall, f1))
                    batch_loss = 0

            self.model.eval()
            with torch.no_grad():
                batch_loss=0
                y_pred, y_list = [], []
                for batch, (article, target, label) in enumerate(val_loader):
                    x = self.tokenizer(list(zip(target, article)), return_tensors='pt', padding='max_length',
                                       max_length=Config["max_length"], truncation=True).to(self.device)
                    y = label.long().to(self.device)
                    output = self.model.forward(x)
                    loss = self.CELoss(output, y)
                    y_pred = output.argmax(dim=1)
                    batch_loss += loss.item()
                    pred_list += y_pred.tolist()
                    y_list += y.tolist()
                pred_list = pred_list[:len(y_list)]
                f1 = f1_score(y_true=y_list, y_pred=pred_list, average='macro')
                pre = precision_score(y_true=y_list, y_pred=pred_list, average='macro')
                recall = recall_score(y_true=y_list, y_pred=pred_list, average='macro')
                print('On test set:')
                print('precision:{:.5f} recall:{:.5f} f1: {:.5f}'.format(pre, recall, f1))
