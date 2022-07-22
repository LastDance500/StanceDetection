"""
BERT model for stance detecting
"""

import os
import torch
import sys
import numpy as np
import warnings

import transformers
from transformers import BertTokenizer, AdamW

from sklearn.metrics import f1_score, precision_score, recall_score
from libs.bert_emb2.model import ComBertModel
from utils.data_during_training import TrainingDataCollector
from libs.bert_emb2.model_train_config import Config

from libs.bert_emb2.preprocessing_and_model import preprocess_bert, load_json


warnings.filterwarnings('ignore')

sys.path.append("../../")
sys.path.append("../")

transformers.logging.set_verbosity_error()
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train):
        # bert
        data_bert = load_json(os.path.join(path, f'bert_emb2/bert_converter.json'))
        bert_matrix = np.load(os.path.join(path, f'bert_emb2/bert_embeddings.npy'))
        stances = load_json(os.path.join(path, f'bert_emb2/final_merged_annotations_correct.json'))
        target_emb = np.load(os.path.join(path, f'bert_emb2/bert_target_embeddings.npy'))
        test_operation=Config['data_name']
        print(test_operation)
        x_train, y_train, x_test, y_test, train_gold_ev, test_gold_ev = preprocess_bert(stances, data_bert, bert_matrix,
                                                                                        stances, target_emb,
                                                                                        op=test_operation,
                                                                                        only_stance=True)
        # embedding
        if train:
            self.s1 = x_train[0]
            self.s2 = x_train[1]
            self.s3 = x_train[2]
            self.s4 = x_train[3]
            self.s5 = x_train[4]
            self.s6 = x_train[5]
            self.article = x_train[6]
            self.target = x_train[7]
            self.y = y_train
        else:
            self.s1 = x_test[0]
            self.s2 = x_test[1]
            self.s3 = x_test[2]
            self.s4 = x_test[3]
            self.s5 = x_test[4]
            self.s6 = x_test[5]
            self.article = x_test[6]
            self.target = x_test[7]
            self.y = y_test

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        s1 = torch.tensor(self.s1[idx])
        s2 = torch.tensor(self.s2[idx])
        s3 = torch.tensor(self.s3[idx])
        s4 = torch.tensor(self.s4[idx])
        s5 = torch.tensor(self.s5[idx])
        s6 = torch.tensor(self.s6[idx])
        article = self.article[idx]
        target = self.target[idx]
        label = torch.tensor(self.y[idx])

        return s1, s2, s3, s4, s5, s6, article, target, label


def get_dataloader(train, batch_size=Config["batch_size"]):
    dataset = Dataset(train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader


class Classifier:

    def __init__(self):
        """
        :param train: train or test
        """
        self.epoch_number = Config["epoch_number"]
        self.device = torch.device(f"cuda:{Config['cuda_index']}" if torch.cuda.is_available() else "cpu")
        self.plot = TrainingDataCollector()
        self.tokenizer = BertTokenizer.from_pretrained(Config["Tokenizer"])
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.model = ComBertModel().to(self.device)

    def train(self, train_loader, val_loader, lr=Config["lr"]):
        # first 20 samples
        cls_list = []
        self.model.eval()
        with torch.no_grad():
            for batch, (s1, s2, s3, s4, s5, s6, article, target, label) in enumerate(val_loader):
                if len(s1) == Config['batch_size']:
                    x = self.tokenizer(list(zip(article, target)), return_tensors='pt', padding='max_length',
                                       truncation=True, max_length=Config["max_length"]).to(self.device)
                    y = label.long().to(self.device)
                    x = x.to(self.device)
                    y = y.to(self.device)
                    s1 = s1.to(self.device)
                    s2 = s2.to(self.device)
                    s3 = s3.to(self.device)
                    s4 = s4.to(self.device)
                    s5 = s5.to(self.device)
                    s6 = s6.to(self.device)
                    if batch <= 20:
                        _, cls = self.model.forward(s1, s2, s3, s4, s5, s6, x, output_emb=True)
                        cls_list.append(cls)
                    else:
                        break

        torch.save(cls_list, os.path.join(path, f'bert_emb2/model/cls_list_bert_random_begin.pt'))

        optimizer = AdamW(self.model.parameters(), lr)
        epoch_list = []
        for epoch in range(self.epoch_number):
            epoch_list.append(epoch + 1)
            pred_list = []
            y_list = []
            batch_loss, total_loss = 0, 0
            self.model.train()
            if not os.path.exists(os.path.join(path, f"model")):
                os.makedirs(os.path.join(path, f"model"))

            for batch, (s1, s2, s3, s4, s5, s6, article, target, label) in enumerate(train_loader):
                if len(s1)== Config['batch_size']:
                    x = self.tokenizer(list(zip(article, target)), return_tensors='pt', padding='max_length',
                                       truncation=True, max_length=Config["max_length"]).to(self.device)
                    y = label.long().to(self.device)
                    x = x.to(self.device)
                    y = y.to(self.device)
                    s1 = s1.to(self.device)
                    s2 = s2.to(self.device)
                    s3 = s3.to(self.device)
                    s4 = s4.to(self.device)
                    s5 = s5.to(self.device)
                    s6 = s6.to(self.device)
                    optimizer.zero_grad()
                    output = self.model.forward(s1, s2, s3, s4, s5, s6, x, output_emb=False).to(self.device)
                    loss = self.CELoss(output, y)
                    batch_loss += loss.data
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    y_pred = output.argmax(dim=1)
                    pred_list += y_pred.tolist()
                    y_list += y.tolist()
                if (batch + 1) % 100 == 0:
                    # pred_list = pred_list[:len(y_list)]
                    f1 = f1_score(y_true=y_list, y_pred=pred_list, average='macro')
                    pre = precision_score(y_true=y_list, y_pred=pred_list, average='macro')
                    recall = recall_score(y_true=y_list, y_pred=pred_list, average='macro')
                    print('[Epoch{} Batch{}] loss:{:.3f} precision:{:.3f} recall:{:.3f} f1:{:.3f}'.format(epoch + 1,
                                                                                                          batch + 1,
                                                                                                          batch_loss,
                                                                                                          pre,
                                                                                                          recall, f1))
                    batch_loss = 0


            y_list=[]
            pred_list = []
            cls_list = []
            self.model.eval()
            with torch.no_grad():
                for batch, (s1, s2, s3, s4, s5, s6, article, target, label) in enumerate(val_loader):
                    if len(s1) == Config['batch_size']:
                        x = self.tokenizer(list(zip(article, target)), return_tensors='pt', padding='max_length',
                                           truncation=True, max_length=Config["max_length"]).to(self.device)
                        y = label.long().to(self.device)
                        x = x.to(self.device)
                        y = y.to(self.device)
                        s1 = s1.to(self.device)
                        s2 = s2.to(self.device)
                        s3 = s3.to(self.device)
                        s4 = s4.to(self.device)
                        s5 = s5.to(self.device)
                        s6 = s6.to(self.device)
                        if batch<=20:
                            output, cls = self.model.forward(s1, s2, s3, s4, s5, s6, x, output_emb=True)
                            output = output.to(self.device)
                            cls_list.append(cls)
                        else:
                            output = self.model.forward(s1, s2, s3, s4, s5, s6, x, output_emb=False).to(self.device)
                        y_pred = output.argmax(dim=1)
                        pred_list += y_pred.tolist()
                        y_list += y.tolist()

            f1 = f1_score(y_true=y_list, y_pred=pred_list, average='macro')
            pre = precision_score(y_true=y_list, y_pred=pred_list, average='macro')
            recall = recall_score(y_true=y_list, y_pred=pred_list, average='macro')
            print('On test set:')
            print('precision:{:.5f} recall:{:.5f} f1: {:.5f}'.format(pre, recall, f1))

            torch.save(pred_list, os.path.join(path, f'bert_emb2/model/pred_list_bert_random.pt'))
            torch.save(cls_list, os.path.join(path, f'bert_emb2/model/cls_list_bert_random_end.pt'))
            torch.save(self.model, os.path.join(path, f'bert_emb2/model/bert_random.pt'))
