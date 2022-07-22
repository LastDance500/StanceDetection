import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .config import config

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(type):
    if type == "train":
        emb = np.load(path + config['train_emb_path'],  allow_pickle=True)
        label = np.load(path + config['train_label_path'],  allow_pickle=True)
    else:
        emb = np.load(path + config['test_emb_path'],  allow_pickle=True)
        label = np.load(path + config['test_label_path'],  allow_pickle=True)

    return emb, label


"""
def process_x(article, title, target, sentence_num):
    try:
        # define the model
        # model = SentenceTransformer('sentence-transformers/nli-bert-base')
        # first item is target, return a list contains strings
        embedding_list = []
        for i in tqdm(range(len(article))):
            body = eval(article[i]) + [title[i]]
            tar = target[i]
            if len(body) < sentence_num:
                body = body + ['' for _ in range(sentence_num-len(body))]
            else:
                body = body[:sentence_num]
            embedding_body = bert_model.encode(body)
            embedding_tar = bert_model.encode(tar)
            embedding = []
            for j in range(len(embedding_body)):
                embedding.append(np.concatenate((embedding_body[j], embedding_tar), axis=None))
            embedding_list.append(embedding)
        return embedding_list
    except Exception as e:
        print(e)

"""


def process_y(label):
    label = np.argmax(label, axis=1)
    return label


def get_dataset_npy(type):
    try:
        np.random.seed(64)
        emb, label = load_data(type)
        for i in range(len(emb)):
            emb[i] = np.array(emb[i])
            # cur = np.array(emb[i])
            # emb[i] = cur/np.sum(cur)

        label = process_y(label)
        random_index = np.random.permutation(len(emb))
        emb = emb[random_index]
        label = label[random_index]
        oh_label= []
        for l in label:
            if l==0:
                oh_label.append([1,0,0,0])
            elif l==1:
                oh_label.append([0,1,0,0])
            elif l==2:
                oh_label.append([0,0,1,0])
            else:
                oh_label.append([0,0,0,1])
        return emb, label, oh_label
    except Exception as e:
        print(e)
