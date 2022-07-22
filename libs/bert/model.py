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

from transformers import BertTokenizer, AdamW, BertForSequenceClassification
from sklearn.metrics import f1_score

from STANDER.utils.data_during_training import TrainingDataCollector, Logger
from STANDER.libs.bert.model_train_config import Config

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

localtime = time.localtime(time.time())
logger_path = str(localtime[0]) + "_" + str(localtime[1])
logger_name = str(localtime[3])

if os.path.exists(os.path.join(path, f'model/{logger_path}/train_{logger_name}.log')):
    logger = Logger(os.path.join(path, f'model/{logger_path}/train_{logger_name}.log'), log_level=logging.INFO,
                    logger_name="Bert_Train").get_log()
elif os.path.exists(path + f'model/{logger_path}'):
    os.mkdir(path + f'/model/{logger_path}')
    logger = Logger(os.path.join(path, f'model/{logger_path}/train_{logger_name}.log'), log_level=logging.INFO,
                    logger_name="Bert_Train").get_log()
else:
    logger = Logger(os.path.join(path, f'model/{logger_path}/train_{logger_name}.log'), log_level=logging.INFO,
                    logger_name="Bert_Train").get_log()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path).fillna("")
        self.article = list(df['article'])
        self.title = list(df['title'])
        self.target = list(df['target'])
        self.label = list(df["label"])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.article[idx], self.target[idx], self.label[idx]


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
        self.tokenizer = BertTokenizer.from_pretrained(Config["Tokenizer"])

        if train:
            self.model = BertForSequenceClassification.from_pretrained(Config["model"], num_labels=4)
            print("train new model by fine-tune")
        else:
            self.model = BertForSequenceClassification.from_pretrained(path + "/model/")
            print("using fine-tune model")
        self.model.to(self.device)

    def predcit(self, article='', target=''):
        self.model.eval()
        x = self.tokenizer(list(zip(article, target)), return_tensors='pt', padding=True, truncation=True).to(
            self.device)
        with torch.no_grad():
            output = self.model(**x)
        return output.logits

    def train(self, train_loader, val_loader, lr=Config["lr"]):
        optimizer = AdamW(self.model.parameters(), lr)
        epoch_list = []
        epoch_acc_train, epoch_loss_train, epoch_acc_test, epoch_loss_test = [], [], [], []
        for epoch in range(self.epoch_number):
            epoch_start_time = time.time()
            total_num, total_loss, total_acc = 0, 0, 0
            val_num, val_loss, val_acc = 0, 0, 0
            self.model.train()
            batch_list, batch_acc_list, batch_loss_list = [], [], []
            epoch_list.append(epoch + 1)

            if not os.path.exists(os.path.join(path, f"model")):
                os.makedirs(os.path.join(path, f"model"))

            for batch, (article, target, label) in enumerate(train_loader):
                x = self.tokenizer(list(zip(article, target)), return_tensors='pt', padding='max_length',
                                   truncation=True, max_length=Config["max_length"]).to(self.device)
                y = label.long().to(self.device)
                cls_emb = self.model.get_input_embeddings()(torch.tensor(self.tokenizer.convert_tokens_to_ids('[CLS]')))
                optimizer.zero_grad()
                output = self.model(**x, labels=y)
                loss = output.loss
                loss.backward()
                optimizer.step()
                y_pred = output.logits.argmax(dim=1)
                acc_num = (y_pred == y).sum().item()
                total_loss += loss.item() * len(y)
                total_acc += acc_num
                total_num += len(y)
                f1 = f1_score(y, y_pred, average='macro')
                if (batch + 1) % 2 == 0:
                    logger.info('[Epoch{} Batch{}] loss:{:.3f} acc:{:.3f} f1:{:.3f}'.format(epoch + 1, batch + 1,
                                                                                            loss.item(),
                                                                                            acc_num / len(y), f1))
                    batch_list.append(batch + 1)
                    batch_acc_list.append(acc_num / len(y))
                    batch_loss_list.append(loss.item())

            self.plot.batch_list, self.plot.accuracy_list, self.plot.loss_list = batch_list, batch_acc_list, \
                                                                                 batch_loss_list
            self.plot.save_path = os.path.join(path, f"model")
            self.plot.plot_accuracy_and_loss_trend(epoch_number=epoch + 1)

            self.model.eval()
            with torch.no_grad():
                for i, (article, target, label) in enumerate(val_loader):
                    x = self.tokenizer(list(zip(article, target)), return_tensors='pt', padding='max_length',
                                       max_length=Config["max_length"], truncation=True).to(self.device)
                    y = label.long().to(self.device)
                    out_put = self.model(**x, labels=y)
                    loss = out_put.loss.item()
                    y_pred = out_put.logits.argmax(dim=1)
                    acc_num = (y_pred == y).sum().item()
                    val_loss += loss * len(y)
                    val_acc += acc_num
                    val_num += len(y)

            logger.info('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' % \
                        (epoch + 1, self.epoch_number, time.time() - epoch_start_time, total_acc / total_num,
                         total_loss / total_num, val_acc / val_num, val_loss / val_num))
            epoch_acc_train.append(total_acc / total_num)
            epoch_loss_train.append(total_loss / total_num)
            epoch_acc_test.append(val_acc / val_num)
            epoch_loss_test.append(val_loss / val_num)

            self.plot.save_path = os.path.join(path, f"model")
            self.plot.accuracy_list_train, self.plot.accuracy_list_test = epoch_acc_train, epoch_acc_test
            self.plot.loss_list_train, self.plot.loss_list_test = epoch_loss_train, epoch_loss_test
            self.plot.epoch_list = epoch_list
            self.plot.plot_whole_train_and_test()
        self.model.save_pretrained(os.path.join(path, f"model"))
        cls_emb = self.model.get_input_embeddings()(torch.tensor(self.tokenizer.convert_tokens_to_ids('[CLS]')))
        torch.save(cls_emb, 'tensor.pt')

    def evaluate(self, val_loader, top_k=None):
        self.model.eval()
        total_acc_num, total_num = 0, 0
        with torch.no_grad():
            for i, (article, target, label) in enumerate(val_loader):
                x = self.tokenizer(list(zip(article, target)), return_tensors='pt', padding='max_length',
                                   max_length=Config["max_length"], truncation=True).to(self.device)
                y = label.long().to(self.device)
                out_put = self.model(**x, labels=y)
                if not top_k:
                    y_pred = out_put.logits.argmax(dim=1)
                    acc_num = (y_pred == y).sum().item()
                else:
                    acc_num = 0
                    y_pred = out_put.logits.argsort(dim=1, descending=True)[:, :top_k]
                    for ind, y_true in enumerate(y):
                        acc_num += 1 if y_true in y_pred[ind] else 0
                total_acc_num += acc_num
                total_num += len(y)
        logger.info('test on %d samples top_%d Acc: %3.6f' % (total_num, top_k, total_acc_num / total_num))
