import os
import torch
import logging
import pandas as pd
import sys
import time
import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
from transformers import BertModel, AlbertModel, BertForSequenceClassification
from STANDER.libs.stance_bert.model_train_config import Config

transformers.logging.set_verbosity_error()
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StanceBerModel(nn.Module):

    def __init__(self):
        super().__init__()
        if Config['new_bert']:
           self.bert = BertModel.from_pretrained(Config["model"])
        else:
           print("===using fine tuned model===")
           self.bert = BertModel.from_pretrained(path + "/model/partA")
        if Config['use_cls']:
           print("===changing cls token===")
           self.cls_emb = torch.load(os.path.join(path, 'model/new_embeddings/pooler_output_embedding_5000_cvs.pt'))
           # self.cls_emb = torch.load(os.path.join(path, 'model/new_embeddings/cls_embedding2_5000_cvs.pt'))
           # self.cls_emb = torch.load(os.path.join(path, 'model/new_embeddings/random_cls_embedding_cvs.pt'))
           with torch.no_grad():
                self.bert.base_model.embeddings.word_embeddings.weight[101] = self.cls_emb[3]
           print("===finished===")
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(768, 128)
        self.linear2 = nn.Linear(128, 4)
        # self.softmax = nn.Softmax(dim=1)
        self.CELoss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.bert(**x).pooler_output
        # torch.save(x, os.path.join(path, 'model/CSV_AET/base_cls_res/sample1.pt'))
        linear_output = self.linear1(x)
        linear_output = self.dropout(linear_output)
        linear_output = F.relu(linear_output)
        linear_output = self.linear2(linear_output)
        # proba = self.softmax(linear_output)
        return linear_output

    def predict(self, x):
        output = self.forward(x)
        return torch.tensor([torch.argmax(output).tolist()])
