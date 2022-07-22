import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.utils.data as data
from model_stander import TwoLayerGCN
from data_preparation_stander import data_setup_stander
from sklearn.metrics import precision_recall_fscore_support
from early_stopping import EarlyStopping
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import BertTokenizer
import transformers
from transformers import BertModel

import warnings
warnings.filterwarnings("ignore")

transformers.logging.set_verbosity_error()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, body, target):
        self.article = list(body)
        self.target = list(target)

    def __len__(self):
        return len(self.article)

    def __getitem__(self, idx):
        a = self.article[idx]
        b = self.target[idx]

        return a, b


if __name__ == '__main__':
    edges_list_tensors, vertices_list_tensors, target, adj, body, target_test_sen, \
    edges_test, vertices_test, target_test, adj_test, body_test, target_sen = data_setup_stander('CVS_AET')

    device = torch.device('cuda:1')

    train_data_loader = data.DataLoader(Dataset(body_test, target_sen), batch_size=1)
    test_data_loader = data.DataLoader(Dataset(body, target_test_sen), batch_size=1)

    embedding_size = len(vertices_list_tensors[0][0][0])
    hidden_dim = 150
    hidden_dim2 = 125
    hidden_dim3 = 50
    learning_rate = 0.1
    dropout = 0.1


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    BERT_model = BertModel.from_pretrained('bert-base-uncased').to(device)


    # model = graph_conv(embedding_size, hidden_dim, hidden_dim2, hidden_dim3, dropout)
    model = TwoLayerGCN(300, 200, 50, 100, dropout=0.1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    samples = len(edges_list_tensors)
    results_all = []
    loss_all = []
    epoch_acc_all = []
    epoch_loss_all = []

    epoch_acc_all_val = []
    epoch_loss_all_val = []
    results_all_val = []

    # number of epochs
    num_epochs = 10

    # early stopping patience
    patience = 5

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='checkpoint.pt')

    for epoch in range(num_epochs):
        loss_sum = 0
        epoch_acc = 0
        results_epoch = []

        loss_sum_val = 0
        epoch_acc_val = 0
        results_epoch_val = []

        model.train()
        optimizer.zero_grad()
        ture_target = []
        for i, (article_sen, target_sen) in enumerate(train_data_loader):
            # optimizer.zero_grad()

            x = tokenizer(list(zip(target_sen, article_sen)), return_tensors='pt', padding='max_length',
                          truncation=True, max_length=512).to(device)

            cls = BERT_model(**x).pooler_output

            result = model(vertices_list_tensors[i][0].to(device), adj[i].to(device), cls)

            results_all.append(result)
            results_epoch.append(result)

            loss = F.cross_entropy(result, target[i].to(device).view(-1))
            loss_all.append(loss)
            loss_sum += float(loss)

            loss.backward()
            optimizer.step()

            ture_target.append(int(target[i]))

        pred_labels = []
        for j in results_epoch:
            predicted_prob, pred_ind = torch.max(F.softmax(j, dim=-1), 1)
            pred_labels.append(int(pred_ind))

        f1 = f1_score(y_true=ture_target, y_pred=pred_labels, average='macro')
        pre = precision_score(y_true=ture_target, y_pred=pred_labels, average='macro')
        recall = recall_score(y_true=ture_target, y_pred=pred_labels, average='macro')

        correct_count = 0
        for i in range(len(target)):
            if pred_labels[i] == target[i]:
                correct_count += 1

        epoch_acc = correct_count / len(target)
        epoch_acc_all.append(epoch_acc)
        epoch_loss_all.append(loss_sum.item() / samples)
        # loss_sum.backward()
        # optimizer.step()

        print('Epoch {:04d} | '.format(epoch) + ' Avg Epoch Loss: {:.4f} | '.format(
            loss_sum.item() / samples) +' train F1: {:.4f} | '.format(2*pre*recall/(pre+recall)) + ' train Pre: {:.4f} | '.format(pre) +
              ' train Rec: {:.4f} | '.format(recall))

        # model.eval()
        # # with torch.no_grad():
        #
        # for i in range(len(edges_val)):
        #     result_val = model(edges_val[i], vertices_val[i], idx_val[i])
        #     results_all_val.append(result_val)
        #     results_epoch_val.append(result_val)
        #
        #     loss_val = F.cross_entropy(result_val, target_val[i].view(-1))
        #     loss_sum_val += loss_val
        #     # acc = binary_accuracy(predictions, batch.label)
        #     # epoch_val_acc += acc.item()
        #
        # pred_labels_val = []
        # for j in results_epoch_val:
        #     predicted_prob_val, pred_ind_val = torch.max(F.softmax(j, dim=-1), 1)
        #     pred_labels_val.append(pred_ind_val)
        #
        # correct_count_val = 0
        # for i in range(len(target_val)):
        #     if pred_labels_val[i] == target_val[i]:
        #         correct_count_val += 1
        #
        # epoch_acc_val = correct_count_val / len(target_val)
        # epoch_acc_all_val.append(epoch_acc_val)
        # epoch_loss_all_val.append(loss_sum_val.item() / len(target_val))
        #
        # print('Epoch {:04d} | '.format(epoch) + ' Avg Epoch Loss: {:.4f} | '.format(
        #     loss_sum.item() / samples) + ' Validation Loss: {:.4f} | '.format(loss_sum_val.item() / len(target_val)) +
        #       ' Epoch Acc: {:.4f} | '.format(epoch_acc) + ' Validation Acc: {:.4f} | '.format(epoch_acc_val))
        #
        # valid_loss = loss_sum_val.item() / len(target_val)
        # early_stopping(valid_loss, model)
        #
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        '''
        Testing the trained model
        '''

        test_loss = 0.0
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))

        model.eval()
        results_all_test = []
        loss_test = []
        loss_sum_test = 0
        true_test_target = []
        for i, (article_sen, target_sen) in enumerate(test_data_loader):
            # optimizer.zero_grad()

            # result = model(edges_list_tensors[i], vertices_list_tensors[i])

            # x = tokenizer(list(zip(target_sen, body)), return_tensors='pt', padding='max_length',
                               # truncation=True, max_length=512)

            # result_test = model(edges_test[i], vertices_test[i])

            x = tokenizer(list(zip(target_sen, article_sen)), return_tensors='pt', padding='max_length',
                          truncation=True, max_length=512).to(device)

            cls = BERT_model(**x).pooler_output

            result_test = model(vertices_test[i][0].to(device), adj_test[i].to(device), cls)

            # result_test = model(vertices_test[i][0], adj_test[i], target_test[i])

            results_all_test.append(result_test)

            loss_t = F.cross_entropy(result_test, target_test[i].to(device).view(-1))
            loss_test.append(loss_t)
            loss_sum_test += loss_t
            true_test_target.append(int(target_test[i]))

        predictions_test = []
        for j in results_all_test:
            _, pred_ind_test = torch.max(F.softmax(j, dim=-1), 1)
            predictions_test.append(int(pred_ind_test))

        f1 = f1_score(y_true=true_test_target, y_pred=predictions_test, average='macro')
        pre = precision_score(y_true=true_test_target, y_pred=predictions_test, average='macro')
        recall = recall_score(y_true=true_test_target, y_pred=predictions_test, average='macro')

        print('On test set:')
        print('precision:{:.5f} recall:{:.5f} f1: {:.5f}'.format(pre, recall, f1))

        correct_test = 0
        for i in range(len(edges_test)):
            if predictions_test[i] == target_test[i]:
                correct_test += 1

        acc_test = correct_test / len(edges_test)

        loss_test_avg = loss_sum_test / len(edges_test)


    # correct_positive = 0
    # correct_negative = 0
    # wrong = 0
    # for i in tqdm(range(len(edges_test))):
    #     if predictions_test[i] == target_test[i] == 1:
    #         correct_positive += 1
    #     elif predictions_test[i] == target_test[i] == 0:
    #         correct_negative += 1
    #     else:
    #         wrong += 1
    #
    # for i in tqdm(range(len(edges_test))):
    #     label = target_test[i]
    #     class_total[label] += 1
    #
    # negative_test_acc = correct_negative / class_total[0]
    # positive_test_acc = correct_positive / class_total[1]
    #
    # print('Test Loss: {:.4f} | '.format(loss_test_avg) + ' Test Acc: {:.4f} | '.format(acc_test))
    # print('Positive test Acc: {:.4f} | '.format(positive_test_acc) + ' Negative test Acc: {:.4f} | '.format(
    #     negative_test_acc))
    #
    # precision_recall_fscore_support(target_test, predictions_test, average='weighted')
