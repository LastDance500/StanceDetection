import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_file(file_name):
    a = torch.load(os.path.join(path, f"STANDER/libs/bert_emb2/model/cls_list_{file_name}_end.pt")).cpu()
    c = a[0][:20]
    return np.array(c)


def plot(matrix, file_name):
    test_data = pd.read_csv(os.path.join(path, f"STANDER/dataset/STANDER/CI_ESRX_test.csv"))
    y = np.array(test_data['label'][:20])
    # target_names = np.array(test_data['stance'][:20])
    target_names = ['refute', 'unrelated', 'support', 'comment']

    pca = PCA(n_components=2)
    X_r = pca.fit(matrix).transform(matrix)

    # plt.figure(facecolor='#DAE3F3')
    ax = plt.axes()
    # ax.set_facecolor('#DAE3F3')

    colors = ["navy", "turquoise", "darkorange", 'red']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
        plt.scatter(
            X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.xlabel('dim1')
    plt.ylabel('dim2')
    plt.show()
    plt.savefig(os.path.join(path, f"STANDER/libs/bert_emb2/model/CSV_AET/cls_list_bert_base_begin.png"))


if __name__ == "__main__":
    # plot cls embeddings
    # with torch.no_grad():
    #     a = np.array(torch.load(os.path.join(path, f"STANDER/libs/bert_emb2/model/cls_embedding_5000_cvs.pt")))
    #     b = np.array(torch.load(os.path.join(path, f"STANDER/libs/bert_emb2/model/cls_embedding2_5000_cvs.pt")))
    #     c = np.array(torch.load(os.path.join(path, f"STANDER/libs/bert_emb2/model/original_cls_embeddings.pt")))
    #     d = np.array(torch.load(os.path.join(path, f"STANDER/libs/bert_emb2/model/random_cls_embedding_cvs.pt")))
    #
    #     e = np.vstack([a, b, c, d])
    # y = np.array([0, 1, 2, 3])
    # target_names = ['cls-transfer1', 'cls-transfer2', 'original-cls', 'random-cls']
    #
    # pca = PCA(n_components=2)
    # X_r = pca.fit(e).transform(e)
    #
    # plt.figure()
    # colors = ["navy", "turquoise", "darkorange", 'red']
    # lw = 2
    #
    # for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
    #     plt.scatter(
    #         X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    #     )
    # plt.legend(loc="best", shadow=False, scatterpoints=1)
    # plt.title("PCA of STANDER dataset")
    # plt.xlabel('dim1')
    # plt.ylabel('dim2')
    # plt.show()
    # plt.savefig(os.path.join(path, f"STANDER/libs/model/embeddings/pca.png"))


    base_sample = read_file('bert_trans')

    plot(base_sample, 'bert_base')

    print(1)
