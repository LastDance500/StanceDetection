"""
Script to train the model
if train == True, then the model will load bert and begin training
if train == False, the model will load the most recent model which saved in the "/model"
"""
import os
from STANDER.libs.bert.model import get_dataloader, Classifier

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    """
    print('--Training one four mergers and prediction--')
    stander_corpus_train = os.path.join(path, f'dataset/STANDER/stander_corpus_train.csv')
    stander_corpus_test = os.path.join(path, f'dataset/STANDER/stander_corpus_test.csv')
    
    train = True
    print("=========Begin training=========")
    train_dataloader = get_dataloader(stander_corpus_train)
    val_dataloader = get_dataloader(stander_corpus_test)
    bert_classifier = Classifier(train=train)
    if train:
        bert_classifier.train(val_dataloader, val_dataloader)
    """

    # CVS_AET prediction
    print("--CVS_AET prediction--")
    stander_corpus_train = os.path.join(path, f'STANDER/dataset/STANDER/CSV_AET_train.csv')
    stander_corpus_test = os.path.join(path, f'STANDER/dataset/STANDER/CSV_AET_test.csv')

    train = True
    print("=========Begin training=========")
    train_dataloader = get_dataloader(stander_corpus_train)
    val_dataloader = get_dataloader(stander_corpus_test)
    bert_classifier = Classifier(train=train)
    if train:
        bert_classifier.train(val_dataloader, val_dataloader)

    """
    # CI_ESRX prediction
    print("--CI_ESRX prediction--")
    stander_corpus_train = os.path.join(path, f'dataset/STANDER/CI_ESRX_train.csv')
    stander_corpus_test = os.path.join(path, f'dataset/STANDER/CI_ESRX_test.csv')

    train = True
    print("=========Begin training=========")
    train_dataloader = get_dataloader(stander_corpus_train)
    val_dataloader = get_dataloader(stander_corpus_test)
    bert_classifier = Classifier(train=train)
    if train:
        bert_classifier.train(val_dataloader, val_dataloader)

    # ANTM_CI prediction
    print("--ANTM_CI prediction--")
    stander_corpus_train = os.path.join(path, f'dataset/STANDER/ANTM_CI_train.csv')
    stander_corpus_test = os.path.join(path, f'dataset/STANDER/ANTM_CI_test.csv')

    train = True
    print("=========Begin training=========")
    train_dataloader = get_dataloader(stander_corpus_train)
    val_dataloader = get_dataloader(stander_corpus_test)
    bert_classifier = Classifier(train=train)
    if train:
        bert_classifier.train(val_dataloader, val_dataloader)


    # AET_HUM prediction
    print("--AET_HUM prediction--")
    stander_corpus_train = os.path.join(path, f'dataset/STANDER/AET_HUM_train.csv')
    stander_corpus_test = os.path.join(path, f'dataset/STANDER/AET_HUM_test.csv')

    train = True
    print("=========Begin training=========")
    train_dataloader = get_dataloader(stander_corpus_train)
    val_dataloader = get_dataloader(stander_corpus_test)
    bert_classifier = Classifier(train=train)
    if train:
        bert_classifier.train(val_dataloader, val_dataloader)
    """