"""
Script to train the model
if train == True, then the model will load bert and begin training
if train == False, the model will load the most recent model which saved in the "/model"
"""
import os
import pandas as pd
from libs.stance_bert.stance_bert_partA import get_dataloader, Classifier

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # CVS_AET prediction
    train_data = os.path.join(path, f'STANDER/dataset/STANDER/augment2/aug_pair_train_5000_CI.csv')
    test_data = os.path.join(path, f'STANDER/dataset/STANDER/augment2/aug_pair_test_5000_CI.csv')

    train = True
    print("=========Begin training=========")
    train_dataloader = get_dataloader(train_data)
    test_dataloader = get_dataloader(test_data)
    bert_classifier = Classifier(train=train)
    if train:
        bert_classifier.train(train_dataloader, test_dataloader)
