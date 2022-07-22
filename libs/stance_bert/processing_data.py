import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle


path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def assign_label(label1, label2):
    label_dict = {(0,0):0, (0,1):1, (0,2):2, (0,3):3, (1,1):4, (1,2):5, (1,3):6, (2,2):7, (2,3):8, (3,3):9}
    combine = (min(label1, label2), max(label1,label2))
    return label_dict[combine]


def generate_pair(df, pair_size, save_path):
    art_title_list_1 = []
    art_title_list_2 = []
    label_list = []
    i = 0
    while True:
        index = np.random.randint(0, len(df), size=2)
        row1 = df.iloc[index[0], :]
        row2 = df.iloc[index[1], :]
        try:
            art1 = row1['title']+row1['article']
            art2 = row2['title']+row2['article']
        except Exception as e:
            continue
        art_title_list_1.append(art1)
        art_title_list_2.append(art2)
        label_list.append(assign_label(label1=row1['label'], label2=row2['label']))
        i += 1
        if i >= 10000:
            break

    new_dict = {'article_title1': art_title_list_1, "article_title2": art_title_list_2, "label": label_list}
    pd.DataFrame(new_dict).to_csv(save_path, index=False)


if __name__ == "__main__":

    AET_test = pd.read_csv(path+"/dataset/STANDER/AET_HUM_test.csv")
    generate_pair(AET_test, 1000, save_path=path+"/dataset/STANDER/AET_HUM_pairs.csv")
    generate_pair(AET_test, 100, save_path=path+"/dataset/STANDER/AET_HUM_pairs_test.csv")

    ANTM_test = pd.read_csv(path+"/dataset/STANDER/ANTM_CI_test.csv")
    generate_pair(AET_test, 1000, save_path=path+"/dataset/STANDER/ANTM_CI_pairs.csv")
    generate_pair(AET_test, 100, save_path=path+"/dataset/STANDER/ANTM_CI_pairs_test.csv")

    AET_test = pd.read_csv(path+"/dataset/STANDER/CI_ESRX_test.csv")
    generate_pair(AET_test, 1000, save_path=path+"/dataset/STANDER/CI_ESRX_pairs.csv")
    generate_pair(AET_test, 100, save_path=path+"/dataset/STANDER/CI_ESRX_pairs_test.csv")

    AET_test = pd.read_csv(path+"/dataset/STANDER/CSV_AET_test.csv")
    generate_pair(AET_test, 1000, save_path=path+"/dataset/STANDER/CSV_AET_pairs.csv")
    generate_pair(AET_test, 100, save_path=path+"/dataset/STANDER/CSV_AET_pairs_test.csv")


