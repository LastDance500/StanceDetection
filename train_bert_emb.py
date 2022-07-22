"""
Script to train the model
if train == True, then the model will load bert and begin training
if train == False, the model will load the most recent model which saved in the "/model"
"""
import os
from libs.bert_emb2.bert_emb import get_dataloader, Classifier

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    print("=========Begin training=========")
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)
    bert_classifier = Classifier()
    bert_classifier.train(train_dataloader, test_dataloader)
