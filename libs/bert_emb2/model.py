import os
import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
from transformers import BertModel
from libs.bert_emb2.model_train_config import Config


transformers.logging.set_verbosity_error()
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ComBertModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device(f"cuda:{Config['cuda_index']}" if torch.cuda.is_available() else "cpu")
        self.method = ''
        self.exchange = False

        if self.method == 'emb':
            self.linear1 = nn.Linear(1536, 136)
            self.linear2 = nn.Linear(1536, 136)
            self.linear3 = nn.Linear(1536, 136)
            self.linear4 = nn.Linear(1536, 136)
            self.linear5 = nn.Linear(1536, 136)
        elif self.method == 'embbert1':
            self.bert = BertModel.from_pretrained(Config["model"])
            print('BertEmb+BERT')
            self.linear1 = nn.Linear(2304, 136)
            self.linear2 = nn.Linear(2304, 136)
            self.linear3 = nn.Linear(2304, 136)
            self.linear4 = nn.Linear(2304, 136)
            self.linear5 = nn.Linear(2304, 136)
        elif self.method == 'embbert2':
            self.bert = BertModel.from_pretrained(Config["model"])
            print('BertEmb+BERT2')
            self.linear1 = nn.Linear(1536, 136)
            self.linear2 = nn.Linear(1536, 136)
            self.linear3 = nn.Linear(1536, 136)
            self.linear4 = nn.Linear(1536, 136)
            self.linear5 = nn.Linear(1536, 136)
        elif self.method == 'embbert3':
            print('BertEmb+BERT3')
            self.bert = BertModel.from_pretrained(Config["model"])
            self.linear1 = nn.Linear(1536, 136)
            self.linear2 = nn.Linear(1536, 136)
            self.linear3 = nn.Linear(1536, 136)
            self.linear4 = nn.Linear(1536, 136)
            self.linear5 = nn.Linear(1536, 136)
            self.linear_cat = nn.Linear(904, 136)
            self.linear_cat1 = nn.Linear(136, 4)
        elif self.method == 'clstrans':
            print("cls-transfer")
            self.linear1 = nn.Linear(768, 4)
            self.bert = BertModel.from_pretrained(Config["model"])
            # self.cls_emb = torch.load(os.path.join(path, 'model/embeddings/random_cls_embedding_cvs.pt'))
            # print("random")
            # self.cls_emb = torch.load(os.path.join(path, 'model/embeddings/cls_embedding2_5000_ci.pt'))
            # print('emb2')
            self.cls_emb = torch.load(os.path.join(path, 'model/embeddings/cls_embedding_5000_ci.pt'))
            print('emb')
            with torch.no_grad():
                self.bert.base_model.embeddings.word_embeddings.weight[101] = self.cls_emb[3]
        else:
            print("only bert")
            self.bert = BertModel.from_pretrained(Config["model"])
            # self.cls_emb = torch.load(os.path.join(path, 'model/embeddings/cls_embedding2_5000_ci.pt'))
            # self.cls_emb = torch.load(os.path.join(path, 'model/embeddings/cls_embedding_5000_ci.pt'))
            self.cls_emb = torch.load(os.path.join(path, 'model/embeddings/random_cls_embedding_cvs.pt'))
            with torch.no_grad():
                self.bert.base_model.embeddings.word_embeddings.weight[101] = self.cls_emb[3]

            self.linear1 = nn.Linear(768, 4)

        self.linear6 = nn.Linear(136, 1)
        self.linear7 = nn.Linear(136, 1)
        self.linear8 = nn.Linear(136, 1)
        self.linear9 = nn.Linear(136, 1)
        self.linear10 = nn.Linear(136, 1)

        self.linear11 = nn.Linear(136, 4)

        # for bertemb2

        self.linear12 = nn.Linear(136+768, 1)
        self.linear13 = nn.Linear(136+768, 1)
        self.linear14 = nn.Linear(136+768, 1)
        self.linear15 = nn.Linear(136+768, 1)
        self.linear16 = nn.Linear(136+768, 1)

        self.linear121 = nn.Linear(136, 1)
        self.linear131 = nn.Linear(136, 1)
        self.linear141 = nn.Linear(136, 1)
        self.linear151 = nn.Linear(136, 1)
        self.linear161 = nn.Linear(136, 1)

        self.linear17 = nn.Linear(136+768, 136)

        self.linear18 = nn.Linear(136, 4)

        self.linear171 = nn.Linear(136, 4)


        # init sentence bert
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.4)
        self.CELoss = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, s1, s2, s3, s4, s5, s6, x, output_emb):

        if self.method =='emb':
            h1 = torch.cat((s2, s1), dim=1)
            h2 = torch.cat((s3, s1), dim=1)
            h3 = torch.cat((s4, s1), dim=1)
            h4 = torch.cat((s5, s1), dim=1)
            h5 = torch.cat((s6, s1), dim=1)

            d1 = self.dropout(self.linear1(h1))
            d2 = self.dropout(self.linear2(h2))
            d3 = self.dropout(self.linear3(h3))
            d4 = self.dropout(self.linear4(h4))
            d5 = self.dropout(self.linear5(h5))

            d = torch.cat([d1, d2, d3, d4, d5], dim=0).reshape(Config['batch_size'], 136, 5)
            # d = d.permute((0, 2, 1))

            a1 = self.sigmoid(self.linear6(d1))
            a2 = self.sigmoid(self.linear7(d2))
            a3 = self.sigmoid(self.linear8(d3))
            a4 = self.sigmoid(self.linear9(d4))
            a5 = self.sigmoid(self.linear10(d5))

            a = torch.cat((a1, a2, a3, a4, a5), dim=1).reshape(Config['batch_size'], 5, 1)
            r = torch.matmul(d, a).reshape(Config['batch_size'], 136)
            output = self.linear11(r)

        elif self.method =='embbert1':
            x = self.bert(**x).pooler_output

            h1 = torch.cat((x, s2, s1), dim=1)
            h2 = torch.cat((x, s3, s1), dim=1)
            h3 = torch.cat((x, s4, s1), dim=1)
            h4 = torch.cat((x, s5, s1), dim=1)
            h5 = torch.cat((x, s6, s1), dim=1)

            # h1 = torch.cat((s2, x, s1), dim=1)
            # h2 = torch.cat((s3, x, s1), dim=1)
            # h3 = torch.cat((s4, x, s1), dim=1)
            # h4 = torch.cat((s5, x, s1), dim=1)
            # h5 = torch.cat((s6, x, s1), dim=1)

            d1 = self.dropout(self.linear1(h1))
            d2 = self.dropout(self.linear2(h2))
            d3 = self.dropout(self.linear3(h3))
            d4 = self.dropout(self.linear4(h4))
            d5 = self.dropout(self.linear5(h5))

            d = torch.cat([d1, d2, d3, d4, d5], dim=0).reshape(Config['batch_size'], 136, 5)
            # d = d.permute((0, 2, 1))

            a1 = self.sigmoid(self.linear6(d1))
            a2 = self.sigmoid(self.linear7(d2))
            a3 = self.sigmoid(self.linear8(d3))
            a4 = self.sigmoid(self.linear9(d4))
            a5 = self.sigmoid(self.linear10(d5))

            a = torch.cat((a1, a2, a3, a4, a5), dim=1).reshape(Config['batch_size'], 5, 1)
            r = torch.matmul(d, a).reshape(Config['batch_size'], 136)
            output = self.linear11(r)

        elif self.method =='embbert2':
            if not self.exchange:
                x = self.bert(**x).pooler_output

                h1 = torch.cat((s2, s1), dim=1)
                h2 = torch.cat((s3, s1), dim=1)
                h3 = torch.cat((s4, s1), dim=1)
                h4 = torch.cat((s5, s1), dim=1)
                h5 = torch.cat((s6, s1), dim=1)

                d1 = self.dropout(self.linear1(h1))
                d2 = self.dropout(self.linear2(h2))
                d3 = self.dropout(self.linear3(h3))
                d4 = self.dropout(self.linear4(h4))
                d5 = self.dropout(self.linear5(h5))

                d1 = torch.cat((d1, x), dim=1)
                d2 = torch.cat((d2, x), dim=1)
                d3 = torch.cat((d3, x), dim=1)
                d4 = torch.cat((d4, x), dim=1)
                d5 = torch.cat((d5, x), dim=1)

                d = torch.cat([d1, d2, d3, d4, d5], dim=0).reshape(Config['batch_size'], 904, 5)
                # d = d.permute((0, 2, 1))

                a1 = self.sigmoid(self.linear12(d1))
                a2 = self.sigmoid(self.linear13(d2))
                a3 = self.sigmoid(self.linear14(d3))
                a4 = self.sigmoid(self.linear15(d4))
                a5 = self.sigmoid(self.linear16(d5))

                a = torch.cat((a1, a2, a3, a4, a5), dim=1).reshape(Config['batch_size'], 5, 1)
                r = torch.matmul(d, a).reshape(Config['batch_size'], 136+768)
                r = self.linear17(r)
                output = self.linear18(F.relu(r))

            else:
                x = self.bert(**x).pooler_output

                h1 = torch.cat((s2, x), dim=1)
                h2 = torch.cat((s3, x), dim=1)
                h3 = torch.cat((s4, x), dim=1)
                h4 = torch.cat((s5, x), dim=1)
                h5 = torch.cat((s6, x), dim=1)

                d1 = self.dropout(self.linear1(h1))
                d2 = self.dropout(self.linear2(h2))
                d3 = self.dropout(self.linear3(h3))
                d4 = self.dropout(self.linear4(h4))
                d5 = self.dropout(self.linear5(h5))

                # d1 = torch.cat((d1, x), dim=1)
                # d2 = torch.cat((d2, x), dim=1)
                # d3 = torch.cat((d3, x), dim=1)
                # d4 = torch.cat((d4, x), dim=1)
                # d5 = torch.cat((d5, x), dim=1)

                d = torch.cat([d1, d2, d3, d4, d5], dim=0).reshape(Config['batch_size'], 136, 5)
                # d = d.permute((0, 2, 1))

                a1 = self.sigmoid(self.linear121(d1))
                a2 = self.sigmoid(self.linear131(d2))
                a3 = self.sigmoid(self.linear141(d3))
                a4 = self.sigmoid(self.linear151(d4))
                a5 = self.sigmoid(self.linear161(d5))

                a = torch.cat((a1, a2, a3, a4, a5), dim=1).reshape(Config['batch_size'], 5, 1)
                r = torch.matmul(d, a).reshape(Config['batch_size'], 136)
                output = self.linear171(r)


        elif self.method == 'embbert3':
            x = self.bert(**x).pooler_output

            h1 = torch.cat((s2, s1), dim=1)
            h2 = torch.cat((s3, s1), dim=1)
            h3 = torch.cat((s4, s1), dim=1)
            h4 = torch.cat((s5, s1), dim=1)
            h5 = torch.cat((s6, s1), dim=1)

            d1 = self.dropout(self.linear1(h1))
            d2 = self.dropout(self.linear2(h2))
            d3 = self.dropout(self.linear3(h3))
            d4 = self.dropout(self.linear4(h4))
            d5 = self.dropout(self.linear5(h5))

            d = torch.cat([d1, d2, d3, d4, d5], dim=0).reshape(Config['batch_size'], 136, 5)
            # d = d.permute((0, 2, 1))

            a1 = self.sigmoid(self.linear6(d1))
            a2 = self.sigmoid(self.linear7(d2))
            a3 = self.sigmoid(self.linear8(d3))
            a4 = self.sigmoid(self.linear9(d4))
            a5 = self.sigmoid(self.linear10(d5))

            a = torch.cat((a1, a2, a3, a4, a5), dim=1).reshape(Config['batch_size'], 5, 1)
            r = torch.matmul(d, a).reshape(Config['batch_size'], 136)

            r = torch.cat((x, r), dim=1)
            r = self.linear_cat(r)
            output = self.linear_cat1(F.relu(r))

        else:
            x = self.bert(**x).pooler_output
            # linear_output = self.linear1(x)
            # linear_output = self.dropout(linear_output)
            # linear_output = F.relu(linear_output)
            output = self.linear1(x)
            if output_emb:
                return output, x
        return output

    def predict(self, s1, s2, s3, s4, s5, s6, x):
        output = self.forward(s1, s2, s3, s4, s5, s6, x)
        return torch.argmax(output, dim=1)

