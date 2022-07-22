import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
bert_model = SentenceTransformer('sentence-transformers/nli-bert-base')

def process_x(article, title, target):
    try:
        # define the model
        # bert_model = SentenceTransformer('sentence-transformers/nli-bert-base')
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


if __name__ == "__main__":
    AET_test = pd.read_csv(path+"/dataset/STANDER/AET_HUM_test.csv")
    ANTM_test = pd.read_csv(path+"/dataset/STANDER/ANTM_CI_test.csv")
    CI_test = pd.read_csv(path+"/dataset/STANDER/CI_ESRX_test.csv")
    CSV_test = pd.read_csv(path+"/dataset/STANDER/CSV_AET_test.csv")

    # index
    AET_ind = AET_test['Unnamed: 0']
    ANTM_ind = ANTM_test['Unnamed: 0']
    CI_ind = CI_test['Unnamed: 0']
    CSV_ind = CSV_test['Unnamed: 0']

    # label
    AET_label = AET_test['label']
    ANTM_label = ANTM_test['label']
    CI_label = CI_test['label']
    CSV_label = CSV_test['label']

    # sentence emb
    df_train = pd.read_csv(path + '/dataset/STANDER/s_stander_corpus_train.csv')
    df_train = df_train.fillna(" ")

    df_test = pd.read_csv(path + '/dataset/STANDER/s_stander_corpus_test.csv')
    df_test = df_test.fillna(" ")

    df_all = pd.concat([df_train, df_test])

    df_all = df_all.set_index(df_all.iloc[:, 0])

    # AET
    article = []
    title = []
    target = []
    for i in AET_ind:
        tmp = df_all.iloc[i]
        article.append(tmp['article'])
        title.append(tmp['title'])
        target.append(tmp['target'])

    AET_X = process_x(article, title, target)
    AET_Y = process_y(AET_label)

    np.save(path + '/dataset/sentence_emb/AET_emb.npy', np.array(AET_X))
    np.save(path + '/dataset/sentence_emb/AET_label.npy', np.array(AET_Y))

    # ANTM
    article = []
    title = []
    target = []
    for i in ANTM_ind:
        tmp = df_all.iloc[i]
        article.append(tmp['article'])
        title.append(tmp['title'])
        target.append(tmp['target'])

    ANTM_X = process_x(article, title, target)
    ANTM_Y = process_y(ANTM_label)

    np.save(path + '/dataset/sentence_emb/ANTM_emb.npy', np.array(ANTM_X))
    np.save(path + '/dataset/sentence_emb/ANTM_label.npy', np.array(ANTM_Y))


    # CI
    article = []
    title = []
    target = []
    for i in CI_ind:
        tmp = df_all.iloc[i]
        article.append(tmp['article'])
        title.append(tmp['title'])
        target.append(tmp['target'])

    CI_X = process_x(article, title, target)
    CI_Y = process_y(CI_label)

    np.save(path + '/dataset/sentence_emb/CI_emb.npy', np.array(CI_X))
    np.save(path + '/dataset/sentence_emb/CI_label.npy', np.array(CI_Y))


    # CSV
    article = []
    title = []
    target = []
    for i in CSV_ind:
        tmp = df_all.iloc[i]
        article.append(tmp['article'])
        title.append(tmp['title'])
        target.append(tmp['target'])

    CSV_X = process_x(article, title, target)
    CSV_Y = process_y(CSV_label)

    np.save(path + '/dataset/sentence_emb/CSV_emb.npy', np.array(CSV_X))
    np.save(path + '/dataset/sentence_emb/CSV_label.npy', np.array(CSV_Y))

    # CSV
    CSV_train_emb = np.concatenate([AET_X, ANTM_X, CI_X])
    CSV_train_label = np.concatenate([AET_Y, ANTM_Y, CI_Y])

    np.save(path + '/dataset/sentence_emb/CSV_train_emb.npy', np.array(CSV_train_emb))
    np.save(path + '/dataset/sentence_emb/CSV_train_label.npy', np.array(CSV_train_label))

    # ANTM
    ANTM_train_emb = np.concatenate([AET_X, CI_X, CSV_X])
    ANTM_train_label = np.concatenate([AET_Y, CI_Y, CSV_Y])

    np.save(path + '/dataset/sentence_emb/ANTM_train_emb.npy', np.array(ANTM_train_emb))
    np.save(path + '/dataset/sentence_emb/ANTM_train_label.npy', np.array(ANTM_train_label))

    # AET
    AET_train_emb = np.concatenate([ANTM_X, CI_X, CSV_X])
    AET_train_label = np.concatenate([ANTM_Y, CI_Y, CSV_Y])

    np.save(path + '/dataset/sentence_emb/AET_train_emb.npy', np.array(AET_train_emb))
    np.save(path + '/dataset/sentence_emb/AET_train_label.npy', np.array(AET_train_label))

    # CI
    CI_train_emb = np.concatenate([AET_X, ANTM_X, CSV_X])
    CI_train_label = np.concatenate([AET_Y, ANTM_Y, CSV_Y])

    np.save(path + '/dataset/sentence_emb/CI_train_emb.npy', np.array(CI_train_emb))
    np.save(path + '/dataset/sentence_emb/CI_train_label.npy', np.array(CI_train_label))
