import os
import numpy as np

from .model_train_config import Config

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(type):
    if type == "train":
        emb = np.load(path + Config['train_emb_path'],  allow_pickle=True)
        label = np.load(path + Config['train_label_path'],  allow_pickle=True)
    else:
        emb = np.load(path + Config['test_emb_path'],  allow_pickle=True)
        label = np.load(path + Config['test_label_path'],  allow_pickle=True)

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


def get_dataset_npy(type, class_number=4):
    try:
        emb, label = load_data(type)
        for i in range(len(emb)):
            emb[i] = np.array(emb[i])
        label = process_y(label)
        random_index = np.random.permutation(len(emb))
        emb = emb[random_index]
        label = label[random_index]

        if class_number == 3:
            index = (label != 0)
            emb = emb[index]
            label = label[index]
            label -= 1
        return emb, label
    except Exception as e:
        print(e)
