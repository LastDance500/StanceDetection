"""
BERT model for stance detecting
"""

import os
import torch
import pandas as pd
import sys
import time

import torch.nn.functional as F

import transformers
from transformers import BertTokenizer, AdamW, BertForSequenceClassification

from sklearn.metrics import f1_score, precision_score, recall_score
from STANDER.libs.bert_emb.combined_model import ComBertModel
from STANDER.utils.data_during_training import TrainingDataCollector
from STANDER.libs.bert_emb.model_train_config import Config
from STANDER.libs.bert_emb.utils_npy import get_dataset_npy

sys.path.append("../../")
sys.path.append("../")

transformers.logging.set_verbosity_error()
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train):
        # bert
        df = pd.read_csv(data_path).fillna("")
        self.article = list(df['article'])
        self.title = list(df['title'])
        self.target = list(df['target'])
        self.label = list(df['label'])

        # embedding
        self.x, self.y = get_dataset_npy(train, Config['output_dim'])
        self.length = Config['batch_length']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        x = F.pad(x, (0, 0, 0, self.length-len(x)))
        return self.title[idx]+self.article[idx], self.target[idx], self.label[idx], x


def get_dataloader(data_path,train, batch_size=Config["batch_size"]):
    dataset = Dataset(data_path, train)
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
        self.tokenizer = BertTokenizer.from_pretrained(Config["Tokenizer"])
        self.CELoss = torch.nn.CrossEntropyLoss()

        if train:
            self.model = ComBertModel()
            print("train new model by fine-tune")
        else:
            self.model = BertForSequenceClassification.from_pretrained(path + "/model/")
            print("using fine-tune model")
        self.model.to(self.device)

    def train(self, train_loader, val_loader, lr=Config["lr"]):
        optimizer = AdamW(self.model.parameters(), lr)
        epoch_list = []
        epoch_acc_train, epoch_loss_train, epoch_acc_test, epoch_loss_test = [], [], [], []
        for epoch in range(self.epoch_number):
            epoch_start_time = time.time()
            total_num, total_loss, total_acc, total_f1, total_pre, total_recall = 0, 0, 0, 0, 0, 0
            val_num, val_loss, val_acc, val_f1, val_pre, val_recall = 0, 0, 0, 0, 0 ,0
            batch_list, batch_acc_list, batch_loss_list = [], [], []
            epoch_list.append(epoch + 1)
            self.model.train()

            if not os.path.exists(os.path.join(path, f"model")):
                os.makedirs(os.path.join(path, f"model"))

            for batch, (article1, article2, label, emb) in enumerate(train_loader):
                x = self.tokenizer(list(zip(article1, article2)), return_tensors='pt', padding='max_length',
                                   truncation=True, max_length=Config["max_length"]).to(self.device)
                y = label.long().to(self.device)
                optimizer.zero_grad()
                output = self.model.forward(x, emb)
                loss = self.CELoss(output, y)
                loss.backward()
                optimizer.step()
                y_pred = output.argmax(dim=1)
                acc_num = (y_pred == y).sum().item()
                total_loss += loss.item() * len(y)
                total_acc += acc_num
                total_num += len(y)
                f1 = f1_score(y_true=y.cpu(), y_pred=y_pred.cpu(), average='macro')
                pre = precision_score(y_true=y.cpu(), y_pred=y_pred.cpu(), average='macro')
                recall = recall_score(y_true=y.cpu(), y_pred=y_pred.cpu(), average='macro')
                total_f1 += f1 * len(y)
                total_pre += pre * len(y)
                total_recall += recall * len(y)
                if (batch + 1) % 20 == 0:
                    print(
                        '[Epoch{} Batch{}] loss:{:.3f} acc:{:.3f} f1:{:.3f} pre:{:.3f} recall:{:.3f}'.format(epoch + 1,
                                                                                                             batch + 1,
                                                                                                             loss.item(),
                                                                                                             acc_num / len(
                                                                                                                 y), f1,
                                                                                                             pre,
                                                                                                             recall))
                    batch_list.append(batch + 1)
                    batch_acc_list.append(acc_num / len(y))
                    batch_loss_list.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                for i, (article1, article2, label, emb) in enumerate(val_loader):
                    x = self.tokenizer(list(zip(article1, article2)), return_tensors='pt', padding='max_length',
                                       max_length=Config["max_length"], truncation=True).to(self.device)
                    y = label.long().to(self.device)
                    output = self.model.forward(x, emb)
                    loss = self.CELoss(output, y)
                    y_pred = output.argmax(dim=1)
                    acc_num = (y_pred == y).sum().item()
                    val_loss += loss * len(y)
                    val_acc += acc_num
                    val_num += len(y)
                    f1 = f1_score(y_true=y.cpu(), y_pred=y_pred.cpu(), average='macro') * len(y)
                    pre = precision_score(y_true=y.cpu(), y_pred=y_pred.cpu(), average='macro') * len(y)
                    recall = recall_score(y_true=y.cpu(), y_pred=y_pred.cpu(), average='macro') * len(y)
                    val_f1 += f1
                    val_pre += pre
                    val_recall += recall

            print(
                '[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f F1: %3.6f| Val Acc: %3.5f F1: %3.5f Pre: %3.5f Recall: %3.5f' % \
                (epoch + 1, self.epoch_number, time.time() - epoch_start_time, total_acc / total_num,
                 total_loss / total_num, total_f1 / total_num, val_acc / val_num, val_f1 / val_num, val_pre/val_num, val_recall/val_num))
            epoch_acc_train.append(total_acc / total_num)
            epoch_loss_train.append(total_loss / total_num)
            epoch_acc_test.append(val_acc / val_num)
            epoch_loss_test.append(val_loss / val_num)
