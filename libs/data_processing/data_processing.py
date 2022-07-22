import os
import json
import pandas as pd
from sklearn.utils import shuffle


path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# read stander_corpus
with open(path+"/dataset/STANDER/stander_corpus.json", "r") as f:
    stander_corpus = json.load(f)
    f.close()

# turn the json file to csv
stance_list = []
article_list = []
title_list = []
target_list = []
for num in stander_corpus.keys():
    if stander_corpus[num]['gold_stance'] != 'unselected':
        stance_list.append(stander_corpus[num]['gold_stance'])
        article_list.append(stander_corpus[num]['sentences'])
        title_list.append(stander_corpus[num]['title'])
        target_list.append(stander_corpus[num]['target'])

for i in range(len(article_list)):
    sentences = article_list[i]
    article = []
    for sentence in sentences:
        article.append(sentence)
    article_list[i] = article

# turn stance to number
stance_num = list(set(stance_list))
stance_dict = {}
for i in range(len(stance_num)):
    stance_dict[stance_num[i]] = i

label_list = []
for i in range(len(stance_list)):
    label_list.append(stance_dict[stance_list[i]])

# save
data_dict = {'article': article_list,
             'title': title_list,
             'target': target_list,
             'stance': stance_list,
             'label': label_list,
             }

data_csv = pd.DataFrame.from_dict(data_dict)
data_csv.to_csv(path+'/dataset/STANDER/s_stander_corpus.csv')
data_csv = data_csv.astype('object')

# data_csv
target = list(set(target_list))
CVS_AET = data_csv.loc[data_csv['target'] == 'CVS (CVS) will merge with Aetna (AET).']
CI_ESRX = data_csv.loc[data_csv['target'] == 'Cigna (CI) will merge with Express Script (ESRX).']
ANTM_CI = data_csv.loc[data_csv['target'] == 'Anthem (ANTM) will merge with Cigna (CI).']
AET_HUM = data_csv.loc[data_csv['target'] == 'Aetna (AET) will merge with Humana (HUM).']


# CVS_AET prediction
frames = [CI_ESRX, ANTM_CI, AET_HUM]
CVS_AET_train = shuffle(pd.concat(frames))
CVS_AET_test = CVS_AET
CVS_AET_train.to_csv(path+'/dataset/STANDER/s_CSV_AET_train.csv')
CVS_AET_test.to_csv(path+'/dataset/STANDER/s_CSV_AET_test.csv')

# CI_ESRX prediction
frames = [CVS_AET, ANTM_CI, AET_HUM]
CI_ESRX_train = shuffle(pd.concat(frames))
CI_ESRX_test = CI_ESRX
CI_ESRX_train.to_csv(path+'/dataset/STANDER/s_CI_ESRX_train.csv')
CI_ESRX_test.to_csv(path+'/dataset/STANDER/s_CI_ESRX_test.csv')

# ANTM_CI prediction
frames = [CVS_AET, CI_ESRX, AET_HUM]
ANTM_CI_train = shuffle(pd.concat(frames))
ANTM_CI_test = ANTM_CI
ANTM_CI_train.to_csv(path+'/dataset/STANDER/s_ANTM_CI_train.csv')
ANTM_CI_test.to_csv(path+'/dataset/STANDER/s_ANTM_CI_test.csv')

# AET_HUM prediction
frames = [CVS_AET, CI_ESRX, ANTM_CI]
AET_HUM_train = shuffle(pd.concat(frames))
AET_HUM_test = AET_HUM
AET_HUM_train.to_csv(path+'/dataset/STANDER/s_AET_HUM_train.csv')
AET_HUM_test.to_csv(path+'/dataset/STANDER/s_AET_HUM_test.csv')

# split dataset into train and test
data_csv = shuffle(data_csv)
data_train = data_csv.iloc[:-500]
data_test = data_csv.iloc[-500:]

data_train.to_csv(path+'/dataset/STANDER/s_stander_corpus_train.csv')
data_test.to_csv(path+'/dataset/STANDER/s_stander_corpus_test.csv')
