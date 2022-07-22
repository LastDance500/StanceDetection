import os
import torch
import logging
import pandas as pd
import sys
import time
import torch
import torch.nn as nn
import transformers
from transformers import BertModel
from STANDER.libs.aug_data_bert.model_train_config import Config

transformers.logging.set_verbosity_error()
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StanceBerModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config["model"])
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(768, 128)
        self.linear2 = nn.Linear(128, 4)
        # self.softmax = nn.Softmax(dim=1)
        # self.CELoss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.bert(**x).pooler_output
        linear_output = self.linear1(x)
        linear_output = self.linear2(linear_output)
        # proba = self.softmax(linear_output)
        return linear_output

    def predict(self, x):
        output = self.forward(x)
        return torch.tensor([torch.argmax(output).tolist()])
