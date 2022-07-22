"""
sentence-BERT model for stance detecting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from STANDER.libs.sentence_bert2.config import config
from sentence_transformers import SentenceTransformer


class VLayer(nn.Module):

    def __init__(self, sentence_dim):
        super(VLayer, self).__init__()
        # self.params = nn.Parameter(torch.randn(config['batch_size'], sentence_dim*2, 1))
        self.params = nn.Parameter(torch.randn(config['batch_size'], sentence_dim*2, 20))

    def forward(self, x):
        if len(x) != config['batch_size']:
            x = F.pad(x, (0, 0, 0, 0, 0, config['batch_size']-len(x)))
        x_cur = x.clone().detach()
        x = torch.sigmoid(torch.matmul(x, self.params))
        # x = torch.transpose(x, 1, 2)
        return torch.mean(torch.matmul(x, x_cur), dim=1)


class BertEmb(nn.Module):

    def __init__(self, sentence_dim, output_dim):
        super().__init__()
        self.epoch_number = config['epoch_number']
        if not config['cpu']:
            self.device = torch.device(f"cuda:{config['cuda_index']}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device('cpu')
        self.sentence_dim = sentence_dim
        self.output_dim = output_dim
        self.class_weights = torch.FloatTensor([3, 6, 2, 3]).to(self.device)
        self.bert_model = SentenceTransformer('sentence-transformers/nli-bert-base')

        # init sentence bert
        self.v_layer = VLayer(sentence_dim)
        # self.fc1 = nn.Linear(sentence_dim*2, 768)
        self.fc1 = nn.Linear(sentence_dim*2, 768)
        self.fc2 = nn.Linear(768, 384)
        self.fc3 = nn.Linear(384, 192)
        self.fc4 = nn.Linear(192, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)
        self.CELoss = nn.CrossEntropyLoss(weight=self.class_weights)


        # cnn
        self.seq_len = 1536
        self.out_size = 100
        self.kernel_1 = 1
        self.stride = 1

        self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)

        # Max pooling layers definition
        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.fc = nn.Linear(100, output_dim)


    def forward(self, x):
        # calculate a
        # x = self.v_layer(x)
        #
        # x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        # x = self.dropout(x)
        #
        # x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        # x = self.dropout(x)
        #
        # x = F.leaky_relu(self.fc3(x), negative_slope=0.1)
        # x = self.dropout(x)
        #
        # x = self.fc4(x)

        # add cnn

        x = self.v_layer(x)

        # Convolution layer 1 is applied
        x1 = self.conv_1(x.reshape(32, 1536, 1))
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1).reshape(32, 100)
        x1 = self.fc(x1)

        return x1

    def predict(self, x):
        output = self.forward(x)
        return torch.argmax(output, dim=1).T

    def loss(self, x, y):
        if len(y) != config['batch_size']:
            y = F.pad(y, (0, config['batch_size']-len(y)))
        output = self.forward(x)
        loss = self.CELoss(output, y)
        return loss
