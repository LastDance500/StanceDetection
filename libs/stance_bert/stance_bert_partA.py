"""
BERT model for stance detecting
"""

import os
import torch
import pandas as pd
import sys
import time

sys.path.append("../../")
sys.path.append("../")
import transformers
from transformers import BertTokenizer, AdamW, BertForSequenceClassification

from sklearn.metrics import f1_score, precision_score, recall_score
from STANDER.libs.stance_bert.Bert_partA import StanceBerModel
from STANDER.utils.data_during_training import TrainingDataCollector, Logger
from STANDER.libs.stance_bert.model_train_config_partA import Config

transformers.logging.set_verbosity_error()
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

localtime = time.localtime(time.time())
logger_path = str(localtime[0]) + "_" + str(localtime[1])
logger_name = str(localtime[3])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path).fillna("")
        self.article1 = list(df['article_title1'])
        self.article2 = list(df['article_title2'])
        self.label = list(df["label"])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.article1[idx], self.article2[idx], self.label[idx]


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
        self.CELoss = torch.nn.CrossEntropyLoss()

        # if train:
        #     self.model = BertForSequenceClassification.from_pretrained(Config["model"], num_labels=Config['num_labels'])
        #     print("train new model by fine-tune")
        # else:
        #     self.model = BertForSequenceClassification.from_pretrained(path + "/model/")
        #     print("using fine-tune model")

        if train:
            self.model = StanceBerModel()
            print("train new model by fine-tune")
        else:
            self.model = BertForSequenceClassification.from_pretrained(path + "/model/")
            print("using fine-tune model")

        cls_emb = self.model.bert.get_input_embeddings()(torch.tensor(self.tokenizer.convert_tokens_to_ids('[CLS]')))
        torch.save(cls_emb, os.path.join(path, 'model/embeddings/original_cls_embeddings.pt'))
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

            for batch, (article1, article2, label) in enumerate(train_loader):
                x = self.tokenizer(list(zip(article2, article1)), return_tensors='pt', padding='max_length',
                                   truncation=True, max_length=Config["max_length"]).to(self.device)
                y = label.long().to(self.device)
                optimizer.zero_grad()
                output = self.model.forward(x)
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
                if (batch + 1) % 50 == 0:
                    print('[Epoch{} Batch{}] loss:{:.3f} acc:{:.3f} f1:{:.3f} pre:{:.3f} recall:{:.3f}'.format(epoch + 1,
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
                for i, (article1, article2, label) in enumerate(val_loader):
                    x = self.tokenizer(list(zip(article2, article1)), return_tensors='pt', padding='max_length',
                                       max_length=Config["max_length"], truncation=True).to(self.device)
                    y = label.long().to(self.device)
                    output = self.model.forward(x)
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
        
        self.model.to("cpu")
        self.model.bert.save_pretrained(os.path.join(path, f"model/partA"))
        cls_emb = self.model.bert.get_input_embeddings()(torch.tensor(self.tokenizer.convert_tokens_to_ids('[CLS]')))
        torch.save(cls_emb, os.path.join(path, 'model/embeddings/cls_embedding2_5000_ci.pt'))
