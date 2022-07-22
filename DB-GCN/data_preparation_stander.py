import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def data_setup(data, raw_data):
    indict = open('glove.840B.300d_dict.pickle', 'rb')
    dictionary = pickle.load(indict)
    embedding_size = 300

    # edge
    dependency_edge = list(data['overall_graph_edges'])
    edges_list_tensors = []

    adj_list_tensors = []
    for news_i in tqdm(range(len(dependency_edge))):
        news = eval(dependency_edge[news_i])
        sentence_edge_tensors = []
        for sen_i in range(len(news)):
            edge = news[sen_i]
            sentence_edge_tensors.append(edge)
            sentence_edge_tensors.append([edge[1], edge[0]])

        edges_list_tensors.append(sentence_edge_tensors)

    # stance label
    stance_list_tensors = []
    stance = list(data['stance'])
    for i in range(len(stance)):
        stance_list_tensors.append(torch.as_tensor(stance[i], dtype=torch.int64))

    # vertices
    tokens = list(data['overall_graph_tokens'])
    vertices_list_tensors = []

    for news_i in range(len(tokens)):
        news_tokens = eval(tokens[news_i])
        news_token_list = []

        matrix_len = len(news_tokens)
        weights_matrix = np.zeros((matrix_len, embedding_size))
        words_found = 0

        for i, word in enumerate(news_tokens):
            try:
                if word == 'ROOT':
                    weights_matrix[i] = weights_matrix[i]
                else:
                    weights_matrix[i] = dictionary[word]
                    words_found += 1
            except KeyError:
                weights_matrix[i] = weights_matrix[i]

        news_token_list.append(torch.as_tensor(weights_matrix, dtype=torch.float))
        # print("Loading {}/{} words from vocab.".format(words_found, len(t)))

        vertices_list_tensors.append(news_token_list)

    for j in tqdm(range(len(edges_list_tensors))):
        edge = edges_list_tensors[j]
        i1 = [m[0] for m in edge]
        i2 = [m[1] for m in edge]
        i = torch.tensor([i1, i2])
        v = torch.tensor([1] * len(i1))
        # a = torch.sparse_coo_tensor(indices=i, values=v, size=[len(i1), len(i1)])
        b = torch.sparse_coo_tensor(indices=i, values=v, size=[len(vertices_list_tensors[j][0]),
                                                               len(vertices_list_tensors[j][0])], dtype=torch.float) \
            .to_dense()
        adj_list_tensors.append(b)

    body_list_tensors = []
    target_list_tensors = []
    for m in range(len(raw_data)):
        try:
            title = raw_data.iloc[m]['title']
            article = raw_data.iloc[m]['article']
            if type(article) != float and type(title) != float:
                body = title + article
            elif type(article) == float:
                body = title
            else:
                body = article
            body_list_tensors.append(body)

            target = raw_data.iloc[m]['target']
            target_list_tensors.append(target)
        except Exception:
            print(1)

    return edges_list_tensors, vertices_list_tensors, stance_list_tensors, adj_list_tensors, body_list_tensors, target_list_tensors


def data_setup_stander(train_name):
    # data = pd.read_csv('data/stander_nospace.csv')
    data = pd.read_csv('data/stander_new3.csv')

    raw_train_data = pd.read_csv(f'data/{train_name}_train.csv')
    raw_test_data = pd.read_csv(f'data/{train_name}_test.csv')

    # target token
    train_data = data.loc[data['target_name'] == train_name]
    train_edges_list_tensors, train_vertices_list_tensors, train_stance_list_tensors, train_adj_list_tensors, \
    train_body_list_tensors, train_target_list_tensors = data_setup(train_data, raw_train_data)

    test_data = data.loc[data['target_name'] != train_name]
    test_edges_list_tensors, test_vertices_list_tensors, test_stance_list_tensors, test_adj_list_tensors, \
    test_body_list_tensors, test_target_list_tensors = data_setup(test_data, raw_test_data)

    return train_edges_list_tensors, train_vertices_list_tensors, train_stance_list_tensors, train_adj_list_tensors, train_body_list_tensors, train_target_list_tensors, \
           test_edges_list_tensors, test_vertices_list_tensors, test_stance_list_tensors, test_adj_list_tensors, test_body_list_tensors, test_target_list_tensors


if __name__ == '__main__':
    data_setup_stander('CI_ESRX')
