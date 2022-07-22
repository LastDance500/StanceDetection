# Stance Detection

This is the code for master thesis of Xiao, in Leiden University. 

## Getting Started

    pip install -r requirements.txt

## Running the tests

1. train_bert_emb.py is the test script BertEmb and CLS-concat BERT
2. train_bert.py is the test script of BERT
3. train_stance_bert.py is the test script of BERT A in CLS-transfer BERT
4. train_stance_bertB.py is the test script of BERT B in CLS-transfer BERT
5. folder DB-GCN is for Dependency-based-GCN

use the following command to test

    python3 filename.py
    

## Dataset

Data is not available so far, please contact the authors of STANDER dataset.

## Built With

  - [Sentence-BERT](https://huggingface.co/bert-base-uncased)
  - [BERT](https://creativecommons.org/)
  

## Authors

  - **Xiao Zhang** - 
    [XiaoZhang](https://github.com/LastDance500)


## Acknowledgments

  - Suzan Verberne and Johan Bos