import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import config

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
bert_model = SentenceTransformer('sentence-transformers/nli-bert-base')


def load_data(type):
    if type == "train":
        df = pd.read_csv(path + config['train_path'])
        df = df.dropna()
    else:
        df = pd.read_csv(path + config['test_path'])
        df = df.dropna()
    return df['article'], df['title'], df['target'], df['label']


def process_x(article, title, target, sentence_num):
    try:
        # define the model
        # model = SentenceTransformer('sentence-transformers/nli-bert-base')
        # first item is target, return a list contains strings
        embedding_list = []
        for i in tqdm(range(len(article))):
            art = []
            for sen in eval(article[i]):
                temp = sen.split(" ")
                if len(temp) >= 25:
                    temp = temp[:25]
                else:
                    temp = temp + [' ']*(25-len(temp))
                string = ' '.join(temp)
                art.append(string)

            tit = []
            temp = title[i].split(" ")
            if len(temp) >= 10:
                temp = temp[:10]
            else:
                temp = temp + [' ']*(10-len(temp))
            tit.append(' '.join(temp))

            body = art + tit
            tar = target[i]
            #if len(body) < sentence_num:
                #body = body + ['' for _ in range(sentence_num-len(body))]
            #else:
                #body = body[:sentence_num]
            embedding_body = bert_model.encode(body)
            embedding_tar = bert_model.encode(tar)
            embedding = []
            for j in range(len(embedding_body)):
                embedding.append(np.concatenate((embedding_body[j], embedding_tar), axis=None))
            embedding_list.append(embedding)
        return embedding_list
    except Exception as e:
        print(e)


def process_y(label):
    label = list(label)
    for i in range(len(label)):
        onehot = [0, 0, 0, 0]
        onehot[label[i]] = 1
        label[i] = onehot
    return label


def get_dataset(type):
    try:
        article, title, target, label = load_data(type)
        X = process_x(article, title, target, config['sentence_num'])
        Y = process_y(label)
        return X, Y
    except Exception as e:
        print(e)


if __name__ == '__main__':

    df_train = pd.read_csv(path + '/dataset/STANDER/s_stander_corpus_train.csv')
    df_train = df_train.fillna(" ")

    df_test = pd.read_csv(path + '/dataset/STANDER/s_stander_corpus_test.csv')
    df_test = df_test.fillna(" ")

    df_all = pd.concat([df_train, df_test])

    """
    article, title, target, label = list(df_test['article']), list(df_test['title']), \
                                    list(df_test['target']), list(df_test['label'])
    X = process_x(article, title, target, sentence_num=25)
    Y = process_y(label)
    df_test['embedding'] = X
    df_test['one_hot_label'] = Y
    np.save(path + '/dataset/STANDER_emb/stander_corpus_test_embedding.npy', np.array(X))
    np.save(path + '/dataset/STANDER_emb/stander_corpus_test_label.npy', np.array(Y))

    article, title, target, label = list(df_train['article']), list(df_train['title']),\
                                    list(df_train['target']), list(df_train['label'])
    X = process_x(article, title, target, sentence_num=25)
    Y = process_y(label)
    df_train['embedding'] = X
    df_train['one_hot_label'] = Y

    np.save(path + '/dataset/STANDER_emb/stander_corpus_train_embedding.npy', np.array(X))
    np.save(path + '/dataset/STANDER_emb/stander_corpus_train_label.npy', np.array(Y))
    """

    embed = np.load(path+'/dataset/STANDER_emb/stander_corpus_train_embedding.npy', allow_pickle=True)
    embed2 = np.load(path+'/dataset/STANDER_emb/stander_corpus_test_embedding.npy', allow_pickle=True)
    emb = np.concatenate([embed, embed2])

    label = np.load(path+'/dataset/STANDER_emb/stander_corpus_train_label.npy', allow_pickle=True)
    label2 = np.load(path+'/dataset/STANDER_emb/stander_corpus_test_label.npy', allow_pickle=True)
    lab = np.concatenate([label, label2])

    df1 = pd.read_csv(path + '/dataset/STANDER/s_ANTM_CI_train.csv')
    df2 = pd.read_csv(path + '/dataset/STANDER/s_ANTM_CI_test.csv')

    all_index = df_all['Unnamed: 0']
    df1_index = np.array(df1['Unnamed: 0'])
    ANTM_CI_train_emb = []
    for i in range(len(all_index)):
        ind = all_index[i]
        if ind in index:
    index = np.array(df1['Unnamed: 0'])
    train_embedding = emb[index]
    train_label = lab[index]

    index = np.array(df2['Unnamed: 0'])
    test_embedding = emb[index]
    test_label = lab[index]

    np.save(path + '/dataset/STANDER_emb/ANTM_CI/train_embedding.npy', train_embedding)
    np.save(path + '/dataset/STANDER_emb/ANTM_CI/train_label.npy', train_label)
    np.save(path + '/dataset/STANDER_emb/ANTM_CI/test_embedding.npy', test_embedding)
    np.save(path + '/dataset/STANDER_emb/ANTM_CI/test_label.npy', test_label)

    # AET_HUM
    df1 = pd.read_csv(path + '/dataset/STANDER/s_AET_HUM_train.csv')
    df2 = pd.read_csv(path + '/dataset/STANDER/s_AET_HUM_test.csv')

    index = np.array(df1['Unnamed: 0'])
    train_embedding = emb[index]
    train_label = lab[index]

    index = np.array(df2['Unnamed: 0'])
    test_embedding = emb[index]
    test_label = lab[index]

    np.save(path + '/dataset/STANDER_emb/AET_HUM/train_embedding.npy', train_embedding)
    np.save(path + '/dataset/STANDER_emb/AET_HUM/train_label.npy', train_label)
    np.save(path + '/dataset/STANDER_emb/AET_HUM/test_embedding.npy', test_embedding)
    np.save(path + '/dataset/STANDER_emb/AET_HUM/test_label.npy', test_label)

    # CI_ESRX
    df1 = pd.read_csv(path + '/dataset/STANDER/s_CI_ESRX_train.csv')
    df2 = pd.read_csv(path + '/dataset/STANDER/s_CI_ESRX_test.csv')

    index = np.array(df1['Unnamed: 0'])
    train_embedding = emb[index]
    train_label = lab[index]

    index = np.array(df2['Unnamed: 0'])
    test_embedding = emb[index]
    test_label = lab[index]

    np.save(path + '/dataset/STANDER_emb/CI_ESRX/train_embedding.npy', train_embedding)
    np.save(path + '/dataset/STANDER_emb/CI_ESRX/train_label.npy', train_label)
    np.save(path + '/dataset/STANDER_emb/CI_ESRX/test_embedding.npy', test_embedding)
    np.save(path + '/dataset/STANDER_emb/CI_ESRX/test_label.npy', test_label)

    # CSV_AET
    df1 = pd.read_csv(path + '/dataset/STANDER/s_CSV_AET_train.csv')
    df2 = pd.read_csv(path + '/dataset/STANDER/s_CSV_AET_test.csv')

    index = np.array(df1['Unnamed: 0'])
    train_embedding = emb[index]
    train_label = lab[index]

    index = np.array(df2['Unnamed: 0'])
    test_embedding = emb[index]
    test_label = lab[index]

    np.save(path + '/dataset/STANDER_emb/CSV_AET/train_embedding.npy', train_embedding)
    np.save(path + '/dataset/STANDER_emb/CSV_AET/train_label.npy', train_label)
    np.save(path + '/dataset/STANDER_emb/CSV_AET/test_embedding.npy', test_embedding)
    np.save(path + '/dataset/STANDER_emb/CSV_AET/test_label.npy', test_label)

