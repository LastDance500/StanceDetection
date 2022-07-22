"""
Script to train the model
if train == True, then the model will load bert and begin training
if train == False, the model will load the most recent model which saved in the "/model"
"""
import os
import pandas as pd
from libs.stance_bert.stance_bert_partB import get_dataloader, Classifier
from libs.stance_bert.model_train_config import Config


path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # CVS_AET prediction
    for i in range(5):
       print("===train CSV_AET data====")
       print(Config)
       print("experiment", i+1)
       train_data = os.path.join(path, f"STANDER/dataset/STANDER/CSV_AET_train.csv")
       test_data = os.path.join(path, f"STANDER/dataset/STANDER/CSV_AET_test.csv")

       train = True
       print("=========Begin training=========")
       train_dataloader = get_dataloader(train_data)
       test_dataloader = get_dataloader(test_data)
       bert_classifier = Classifier(train=train)
       if train:
          bert_classifier.train(train_dataloader, test_dataloader)
          # bert_classifier.test(test_dataloader)

for i in range(5):
   print("===train CI_ESRX data====")
   print(Config)
   print("experiment", i + 1)
   train_data = os.path.join(path, f"STANDER/dataset/STANDER/CI_ESRX_train.csv")
   test_data = os.path.join(path, f"STANDER/dataset/STANDER/CI_ESRX_test.csv")

   train = True
   print("=========Begin training=========")
   train_dataloader = get_dataloader(train_data)
   test_dataloader = get_dataloader(test_data)
   bert_classifier = Classifier(train=train)
   if train:
      bert_classifier.train(train_dataloader, test_dataloader)
      # bert_classifier.test(test_dataloader)

   for i in range(5):
      print("===train ANTM_CI data====")
      print(Config)
      print("experiment", i + 1)
      train_data = os.path.join(path, f"STANDER/dataset/STANDER/ANTM_CI_train.csv")
      test_data = os.path.join(path, f"STANDER/dataset/STANDER/ANTM_CI_test.csv")

      train = True
      print("=========Begin training=========")
      train_dataloader = get_dataloader(train_data)
      test_dataloader = get_dataloader(test_data)
      bert_classifier = Classifier(train=train)
      if train:
         bert_classifier.train(train_dataloader, test_dataloader)
         # bert_classifier.test(test_dataloader)

   for i in range(5):
      print("===train AET_HUM data====")
      print(Config)
      print("experiment", i + 1)
      train_data = os.path.join(path, f"STANDER/dataset/STANDER/AET_HUM_train.csv")
      test_data = os.path.join(path, f"STANDER/dataset/STANDER/AET_HUM_test.csv")

      train = True
      print("=========Begin training=========")
      train_dataloader = get_dataloader(train_data)
      test_dataloader = get_dataloader(test_data)
      bert_classifier = Classifier(train=train)
      if train:
         bert_classifier.train(train_dataloader, test_dataloader)
         # bert_classifier.test(test_dataloader)
