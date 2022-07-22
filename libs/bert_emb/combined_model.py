import os
import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
from transformers import BertModel
from STANDER.libs.bert_emb.model_train_config import Config


transformers.logging.set_verbosity_error()
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VLayer(nn.Module):

    def __init__(self, sentence_dim):
        super(VLayer, self).__init__()
        self.device = torch.device(f"cuda:{Config['cuda_index']}" if torch.cuda.is_available() else "cpu")
        self.params = nn.Parameter(torch.randn(Config['batch_size'], sentence_dim*2, 1))

    def forward(self, x):
        try:
            x = x.to(self.device)
            if len(x) != Config['batch_size']:
                x = F.pad(x, (0, 0, 0, 0, 0, Config['batch_size']-len(x)))
            x_cur = x.clone().detach()
            x = torch.sigmoid(torch.matmul(x, self.params))
            x = torch.transpose(x, 1, 2)
            x = torch.matmul(x, x_cur)
        except Exception as e:
            print(e)

        return x


class ComBertModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device(f"cuda:{Config['cuda_index']}" if torch.cuda.is_available() else "cpu")
        self.bert = BertModel.from_pretrained(Config["model"])
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(768*3, 768)
        self.linear2 = nn.Linear(768, 128)
        self.linear3 = nn.Linear(128, 4)

        #
        self.sentence_dim = Config['sentence_dim']
        self.output_dim = Config['output_dim']

        # init sentence bert
        self.v_layer = VLayer(self.sentence_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)
        self.CELoss = nn.CrossEntropyLoss()

    def forward(self, x, emb_x):
        x = self.bert(**x).pooler_output

        # calculate a
        emb_x = self.v_layer(emb_x)

        emb_x = torch.reshape(emb_x, (emb_x.shape[0], emb_x.shape[2]))
        if len(x) != Config['batch_size']:
            emb_x = emb_x[:len(x)]
        x = torch.cat((emb_x, x), dim=1)

        linear_output = self.linear1(x)
        linear_output = F.relu(linear_output)
        linear_output = self.linear2(linear_output)
        linear_output = F.relu(linear_output)
        linear_output = self.linear3(linear_output)

        return linear_output

    def predict(self, x, x_emb):
        output = self.forward(x, x_emb)
        return torch.tensor([torch.argmax(output).tolist()])
