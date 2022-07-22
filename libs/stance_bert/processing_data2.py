import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.utils import shuffle

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 0: refute
# 1: unrelated
# 2: support
# 3: comment

def assign_label_1(label1, label2):
    label_dict = {(0, 0): 2, (0, 1): 1, (0, 2): 0, (0, 3): 0,
                  (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1,
                  (2, 0): 0, (2, 1): 1, (2, 2): 2, (2, 3): 0,
                  (3, 0): 3, (3, 1): 1, (3, 2): 3, (3, 3): 2}
    combine = (label1, label2)
    return label_dict[combine]


def assign_label_2(label1, label2):
    label_dict = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3,
                  (1, 1): 4, (1, 2): 5, (1, 3): 6,
                  (2, 2): 7, (2, 3): 8,
                  (3, 3): 9}
    combine = (min((label1, label2)), max((label1, label2)))
    return label_dict[combine]


def generate_pair(df, pair_size, save_path1, save_path2):
    art_title_list_1 = []
    art_title_list_2 = []
    label_list1 = []
    label_list2 = []
    i = 0
    while True:
        index = np.random.randint(0, len(df), size=2)
        row1 = df.iloc[index[0], :]
        row2 = df.iloc[index[1], :]
        try:
            art1 = row1['title'] + row1['article']
            art2 = row2['title'] + row2['article']
        except Exception as e:
            continue
        art_title_list_1.append(art1)
        art_title_list_2.append(art2)
        label_list1.append(assign_label_1(label1=row1['label'], label2=row2['label']))
        label_list2.append(assign_label_2(label1=row1['label'], label2=row2['label']))
        i += 1
        if i >= pair_size:
            break

    new_dict1 = {'article_title1': art_title_list_1, "article_title2": art_title_list_2, "label": label_list1}
    pd.DataFrame(new_dict1).to_csv(save_path1, index=False)

    new_dict2 = {'article_title1': art_title_list_1, "article_title2": art_title_list_2, "label": label_list2}
    pd.DataFrame(new_dict2).to_csv(save_path2, index=False)


if __name__ == "__main__":
    # only run AET
    total_num = 5000
    AET_test = pd.read_csv(path+"/dataset/STANDER/AET_HUM_test.csv")
    generate_pair(AET_test, int(0.3*total_num), save_path1=path+"/dataset/STANDER/augment/AET_HUM_pairs.csv",
                  save_path2=path+"/dataset/STANDER/augment2/AET_HUM_pairs.csv")
    generate_pair(AET_test, int(0.15*total_num), save_path1=path+"/dataset/STANDER/augment/AET_HUM_pairs_test.csv",
                  save_path2=path+"/dataset/STANDER/augment2/AET_HUM_pairs_test.csv")

    ANTM_test = pd.read_csv(path+"/dataset/STANDER/ANTM_CI_test.csv")
    generate_pair(ANTM_test, int(0.3*total_num), save_path1=path+"/dataset/STANDER/augment/ANTM_CI_pairs.csv",
                  save_path2=path+"/dataset/STANDER/augment2/ANTM_CI_pairs.csv")
    generate_pair(ANTM_test, int(0.15*total_num), save_path1=path+"/dataset/STANDER/augment/ANTM_CI_pairs_test.csv",
                  save_path2=path+"/dataset/STANDER/augment2/ANTM_CI_pairs_test.csv")

    CI_test = pd.read_csv(path+"/dataset/STANDER/CI_ESRX_test.csv")
    generate_pair(CI_test, int(0.2*total_num), save_path1=path+"/dataset/STANDER/augment/CI_ESRX_pairs.csv",
                  save_path2=path+"/dataset/STANDER/augment2/CI_ESRX_pairs.csv")
    generate_pair(CI_test, int(0.08*total_num), save_path1=path+"/dataset/STANDER/augment/CI_ESRX_pairs_test.csv",
                  save_path2=path+"/dataset/STANDER/augment2/CI_ESRX_pairs_test.csv")

    CSV_test = pd.read_csv(path+"/dataset/STANDER/CSV_AET_test.csv")
    generate_pair(CSV_test, int(0.2*total_num), save_path1=path+"/dataset/STANDER/augment/CSV_AET_pairs.csv",
                  save_path2=path+"/dataset/STANDER/augment2/CSV_AET_pairs.csv")
    generate_pair(CSV_test, int(0.1*total_num), save_path1=path+"/dataset/STANDER/augment/CSV_AET_pairs_test.csv",
                  save_path2=path+"/dataset/STANDER/augment2/CSV_AET_pairs_test.csv")


    # method1
    train_data1 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment/AET_HUM_pairs.csv'))
    train_data2 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment/ANTM_CI_pairs.csv'))
    train_data3 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment/CI_ESRX_pairs.csv'))
    train_data4 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment/CSV_AET_pairs.csv'))

    test_data1 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment/AET_HUM_pairs_test.csv'))
    test_data2 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment/ANTM_CI_pairs_test.csv'))
    test_data3 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment/CI_ESRX_pairs_test.csv'))
    test_data4 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment/CSV_AET_pairs_test.csv'))


    # method2
    train_data5 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment2/AET_HUM_pairs.csv'))
    train_data6 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment2/ANTM_CI_pairs.csv'))
    train_data7 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment2/CI_ESRX_pairs.csv'))
    train_data8 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment2/CSV_AET_pairs.csv'))

    test_data5 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment2/AET_HUM_pairs_test.csv'))
    test_data6 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment2/ANTM_CI_pairs_test.csv'))
    test_data7 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment2/CI_ESRX_pairs_test.csv'))
    test_data8 = pd.read_csv(os.path.join(path, f'dataset/STANDER/augment2/CSV_AET_pairs_test.csv'))


    # AET
    train_data = pd.concat([train_data2, train_data3,
                            train_data4])
    test_data = pd.concat([test_data2, test_data3,
                            test_data4])
    train_data = shuffle(train_data)
    test_data = shuffle(test_data)

    train_data.to_csv(path+f'/dataset/STANDER/augment/aug_pair_train_{total_num}_AET.csv', index=False)
    test_data.to_csv(path+f'/dataset/STANDER/augment/aug_pair_test_{total_num}_AET.csv', index=False)

    train_data = pd.concat([train_data6, train_data7,
                            train_data8])
    test_data = pd.concat([test_data6, test_data7,
                            test_data8])
    train_data = shuffle(train_data)
    test_data = shuffle(test_data)

    train_data.to_csv(path+f'/dataset/STANDER/augment2/aug_pair_train_{total_num}_AET.csv', index=False)
    test_data.to_csv(path+f'/dataset/STANDER/augment2/aug_pair_test_{total_num}_AET.csv', index=False)

    # ANTM
    train_data = pd.concat([train_data1, train_data3,
                            train_data4])
    test_data = pd.concat([test_data1, test_data3,
                            test_data4])
    train_data = shuffle(train_data)
    test_data = shuffle(test_data)

    train_data.to_csv(path+f'/dataset/STANDER/augment/aug_pair_train_{total_num}_ANTM.csv', index=False)
    test_data.to_csv(path+f'/dataset/STANDER/augment/aug_pair_test_{total_num}_ANTM.csv', index=False)

    train_data = pd.concat([train_data5, train_data7,
                            train_data8])
    test_data = pd.concat([test_data5, test_data7,
                            test_data8])
    train_data = shuffle(train_data)
    test_data = shuffle(test_data)

    train_data.to_csv(path+f'/dataset/STANDER/augment2/aug_pair_train_{total_num}_ANTM.csv', index=False)
    test_data.to_csv(path+f'/dataset/STANDER/augment2/aug_pair_test_{total_num}_ANTM.csv', index=False)

    # CI
    train_data = pd.concat([train_data1, train_data2,
                            train_data4])
    test_data = pd.concat([test_data1, test_data2,
                            test_data4])
    train_data = shuffle(train_data)
    test_data = shuffle(test_data)

    train_data.to_csv(path+f'/dataset/STANDER/augment/aug_pair_train_{total_num}_CI.csv', index=False)
    test_data.to_csv(path+f'/dataset/STANDER/augment/aug_pair_test_{total_num}_CI.csv', index=False)

    train_data = pd.concat([train_data5, train_data6,
                            train_data8])
    test_data = pd.concat([test_data5, test_data6,
                            test_data8])
    train_data = shuffle(train_data)
    test_data = shuffle(test_data)

    train_data.to_csv(path+f'/dataset/STANDER/augment2/aug_pair_train_{total_num}_CI.csv', index=False)
    test_data.to_csv(path+f'/dataset/STANDER/augment2/aug_pair_test_{total_num}_CI.csv', index=False)

    # CVS
    train_data = pd.concat([train_data1, train_data2,
                            train_data3])
    test_data = pd.concat([test_data1, test_data2,
                            test_data3])
    train_data = shuffle(train_data)
    test_data = shuffle(test_data)

    train_data.to_csv(path+f'/dataset/STANDER/augment/aug_pair_train_{total_num}_CVS.csv', index=False)
    test_data.to_csv(path+f'/dataset/STANDER/augment/aug_pair_test_{total_num}_CVS.csv', index=False)


    train_data = pd.concat([train_data5, train_data6,
                            train_data7])
    test_data = pd.concat([test_data5, test_data6,
                            test_data7])
    train_data = shuffle(train_data)
    test_data = shuffle(test_data)

    train_data.to_csv(path+f'/dataset/STANDER/augment2/aug_pair_train_{total_num}_CVS.csv', index=False)
    test_data.to_csv(path+f'/dataset/STANDER/augment2/aug_pair_test_{total_num}_CVS.csv', index=False)
    print(1)

