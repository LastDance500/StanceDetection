import numpy

numpy.random.seed(42)

import tensorflow as tf
# import tensorflow_hub as hub

# url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# embed = hub.Module(url)

from keras.layers import *
from keras.models import Model
from keras.callbacks import *
from keras.metrics import binary_accuracy, categorical_accuracy
import keras.backend as K
# from keras.utils import plot_model

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report

# from use import UniversalEmbeddingLayer

# import useful variables
# from autoassign import initializer


SEED = 42


def dump_json(content, filename):
    with open(filename, "w") as fout:
        json.dump(content, fout, indent=4)


def load_json(filename):
    with open(filename, "r") as fin:
        return json.load(fin)


class NewMetrics(Callback):

    def __init__(self, train_data, train_labels, dev_data, dev_labels, train_gold_ev, test_gold_ev, only_stance,
                 reversed_stance_dict, number_sents=5):
        self.train_data = train_data
        self.train_labels = train_labels
        self.dev_data = dev_data
        self.dev_labels = dev_labels
        self.train_gold_ev = train_gold_ev
        self.test_gold_ev = test_gold_ev
        self.only_stance = only_stance
        self.reversed_stance_dict = reversed_stance_dict
        self.number_sents = number_sents

    def on_train_begin(self, logs={}):
        self.precisions, self.recalls, self.f1_scores = [], [], []
        self.val_precisions, self.val_recalls, self.val_f1_scores = [], [], []

    def evidence_macro_precision(self, gold, pred):
        this_precision = 0.0
        this_precision_hits = 0.0

        for prediction in pred:
            if prediction in gold:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    def evidence_prec(self, golds, preds):
        macro_precision = 0
        macro_precision_hits = 0
        print(len(golds), len(preds))

        for i in range(len(golds)):
            pred = preds[i]
            predicted_evidence = [i for i in range(len(pred)) if pred[i] != 0]
            macro_prec = self.evidence_macro_precision(golds[i], predicted_evidence)
            macro_precision += macro_prec[0]
            macro_precision_hits += macro_prec[1]

        pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
        return pr

    def evidence_macro_recall(self, gold, pred):
        if len(gold) == 0:
            return 1.0, 1.0
        for p in pred:
            if all([item in pred for item in gold]):
                # We only want to score complete groups of evidence. Incomplete groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0

    def evidence_rec(self, golds, preds):
        macro_recall = 0
        macro_recall_hits = 0

        for i in range(len(golds)):
            pred = preds[i]
            predicted_evidence = [i for i in range(len(pred)) if pred[i] != 0]
            macro_rec = self.evidence_macro_recall(golds[i], predicted_evidence)
            macro_recall += macro_rec[0]
            macro_recall_hits += macro_rec[1]

        rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0
        return rec

    def on_epoch_end(self, batch, logs={}):

        y_pred = self.model.predict(self.train_data)
        _y_pred = y_pred
        gold = self.train_labels

        if not self.only_stance:
            print("train_gold_ev.shape, y_pred.shape", len(self.train_gold_ev), len(y_pred[1]))
            pred_evidences = [[numpy.round(y_pred[i][sample_no])[0] for i in range(1, self.number_sents + 1)] \
                              for sample_no in range(len(self.train_gold_ev))]

            er_r, er_p = self.evidence_rec(self.train_gold_ev, pred_evidences), \
                         self.evidence_prec(self.train_gold_ev, pred_evidences)
            er_f = 2.0 * er_p * er_r / (er_p + er_r)

            _y_pred = y_pred[0]
            gold = gold[0]

        _y_pred = [self.reversed_stance_dict[int(numpy.argmax(i))] for i in _y_pred]
        gold = [self.reversed_stance_dict[int(numpy.argmax(i))] for i in gold]

        _f1 = f1_score(gold, _y_pred, average="macro")
        _recall = recall_score(gold, _y_pred, average="macro")
        _precision = precision_score(gold, _y_pred, average="macro")

        y_val_pred = self.model.predict(self.dev_data)
        _y_val_pred = y_val_pred
        gold_pred = self.dev_labels

        if not self.only_stance:
            pred_evidences = [[numpy.round(y_pred[i][sample_no])[0] for i in range(1, self.number_sents + 1)] \
                              for sample_no in range(len(self.test_gold_ev))]
            val_er_r, val_er_p = self.evidence_rec(self.test_gold_ev, pred_evidences), self.evidence_prec(
                self.test_gold_ev, pred_evidences)
            val_er_f = 2.0 * val_er_p * val_er_r / (val_er_p + val_er_r)

            _y_val_pred = y_val_pred[0]
            gold_pred = gold_pred[0]

        _y_val_pred = [self.reversed_stance_dict[int(numpy.argmax(i))] for i in _y_val_pred]
        gold_pred = [self.reversed_stance_dict[int(numpy.argmax(i))] for i in gold_pred]

        _val_f1 = f1_score(gold_pred, _y_val_pred, average="macro")
        _val_recall = recall_score(gold_pred, _y_val_pred, average="macro")
        _val_precision = precision_score(gold_pred, _y_val_pred, average="macro")

        self.precisions.append(_precision)
        self.recalls.append(_recall)
        self.f1_scores.append(_f1)
        self.val_precisions.append(_val_precision)
        self.val_recalls.append(_val_recall)
        self.val_f1_scores.append(_val_f1)

        if not self.only_stance:
            print("EVIDENCE RETRIEVAL RESULTS:")
            print("TRAIN")
            print("F1 score : ", er_f)
            print("Recall   : ", er_r)
            print("Precision: ", er_p)

            print("OTHER")
            print("F1 score : ", val_er_f)
            print("Recall   : ", val_er_r)
            print("Precision: ", val_er_p)
        print("")
        print("STANCE DETECTION")
        print(" — f1: ", round(_f1, 4), " — precision: ", round(_precision, 4), " — recall", round(_recall, 4), \
              " — val_f1: ", round(_val_f1, 4), " — val_precision: ", round(_val_precision, 4), " — val_recall",
              round(_val_recall, 4))

        print(confusion_matrix(gold_pred, _y_val_pred))
        print(classification_report(gold_pred, _y_val_pred))


class WingOsModel():
    def __init__(self, \
                 experiment_name, \
                 encoder, \
                 only_stance=False, \
                 voc_size=1000, \
                 max_sent_length=25, \
                 epochs=70, \
                 batch_size=32, \
                 finetune_embeddings=False, \
                 embedding_size=300, \
                 use_embedding_embeddings=True, \
                 USE_embed_size=512, \
                 BERT_embed_size=768, \
                 BERT_dense=True, \
                 no_classes=3, \
                 input_vocabulary=None, \
                 use_BiLSTM_pretrained_embeddings=True, \
                 pretrained_embedding_file="glove.6B.300d.txt", \
                 pretrained_embeddings_type="glove", \
                 LSTM_dropout=0.2, \
                 LSTM_rec_dropout=0.2, \
                 bilstm_dense=68, \
                 USE_dense=True, \
                 use_dense_size=256, \
                 stance_dict={"support": 0, "refute": 1, "comment": 2, "unrelated": 3}, \
                 reversed_stance_dict={0: "support", 1: "refute", 2: "comment", 3: "unrelated"}):

        self.experiment_name = experiment_name
        self.encoder = encoder
        self.only_stance = only_stance
        self.voc_size = voc_size
        self.max_sent_length = max_sent_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.finetune_embeddings = finetune_embeddings
        self.embedding_size = embedding_size
        self.use_embedding_embeddings = use_embedding_embeddings
        self.USE_embed_size = USE_embed_size
        self.BERT_embed_size = BERT_embed_size
        self.BERT_dense = BERT_dense
        self.no_classes = no_classes
        self.input_vocabulary = input_vocabulary
        self.use_BiLSTM_pretrained_embeddings = use_BiLSTM_pretrained_embeddings
        self.pretrained_embedding_file = pretrained_embedding_file
        self.pretrained_embeddings_type = pretrained_embeddings_type
        self.LSTM_dropout = LSTM_dropout
        self.LSTM_rec_dropout = LSTM_rec_dropout
        self.bilstm_dense = bilstm_dense
        self.USE_dense = USE_dense
        self.use_dense_size = use_dense_size
        self.stance_dict = stance_dict
        self.reversed_stance_dict = reversed_stance_dict

    def get_pretrained_embedding_matrix(self):
        """
        Load pretrained embedding matrix from file.
        """

        embeddings_index = {}

        # read the file
        pretr_file = self.pretrained_embedding_file

        print("Loading embeddings: ", pretr_file)

        if self.pretrained_embeddings_type == "glove":
            with open(pretr_file) as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = numpy.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs

            vocab_size = (len(self.input_vocabulary.keys()))
            pretr_embedding_matrix = numpy.zeros((vocab_size, self.embedding_size))

            for word, i in self.input_vocabulary.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    pretr_embedding_matrix[i] = embedding_vector

        return pretr_embedding_matrix

    def build_model(self):

        if self.encoder == "BERT":
            _input_target = Input(shape=(self.BERT_embed_size,))
            _input_sentence_1 = Input(shape=(self.BERT_embed_size,))
            _input_sentence_2 = Input(shape=(self.BERT_embed_size,))
            _input_sentence_3 = Input(shape=(self.BERT_embed_size,))
            _input_sentence_4 = Input(shape=(self.BERT_embed_size,))
            _input_sentence_5 = Input(shape=(self.BERT_embed_size,))

            h_1 = concatenate([_input_target, _input_sentence_1])
            h_2 = concatenate([_input_target, _input_sentence_2])
            h_3 = concatenate([_input_target, _input_sentence_3])
            h_4 = concatenate([_input_target, _input_sentence_4])
            h_5 = concatenate([_input_target, _input_sentence_5])

            if self.BERT_dense:
                h_1 = Dense(self.use_dense_size)(h_1)
                h_2 = Dense(self.use_dense_size)(h_2)
                h_3 = Dense(self.use_dense_size)(h_3)
                h_4 = Dense(self.use_dense_size)(h_4)
                h_5 = Dense(self.use_dense_size)(h_5)

            h_1 = Dropout(0.4)(h_1)
            h_2 = Dropout(0.4)(h_2)
            h_3 = Dropout(0.4)(h_3)
            h_4 = Dropout(0.4)(h_4)
            h_5 = Dropout(0.4)(h_5)

        # concatenate sentence representations

        # TASK 1

        alpha_1 = Dense(1, activation="sigmoid", name="alpha_1")(h_1)
        alpha_2 = Dense(1, activation="sigmoid", name="alpha_2")(h_2)
        alpha_3 = Dense(1, activation="sigmoid", name="alpha_3")(h_3)
        alpha_4 = Dense(1, activation="sigmoid", name="alpha_4")(h_4)
        alpha_5 = Dense(1, activation="sigmoid", name="alpha_5")(h_5)

        alphas = concatenate([alpha_1, alpha_2, alpha_3, alpha_4, alpha_5])
        print("alphas.shape", alphas.shape)

        # TASK 2

        print('h_1.shape', h_1.shape)
        sents = concatenate([h_1, h_2, h_3, h_4, h_5], axis=-1)
        print("sents.shape", sents.shape)

        if self.BERT_dense or self.USE_dense:
            sents = Reshape((5, self.use_dense_size))(sents)
        else:
            sents = Reshape((5, self.bilstm_dense * 2))(sents)

        print("sents.shape", sents.shape)
        sents = Permute((2, 1))(sents)
        print("sents.shape", sents.shape)
        e = Dot(axes=-1)([sents, alphas])
        print("e.shape", e.shape)

        stance_probs = Dense(self.no_classes, activation="softmax", name="stance_probs")(e)

        # --- compile model ---
        # define variables
        inputs = [_input_target, _input_sentence_1, _input_sentence_2, _input_sentence_3, _input_sentence_4,
                  _input_sentence_5]

        if not self.only_stance:
            outputs = [stance_probs, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5]
            self.model = Model(inputs=inputs, outputs=outputs)
            self.loss = {"stance_probs": "categorical_crossentropy", "alpha_1": "binary_crossentropy",
                         "alpha_2": "binary_crossentropy", "alpha_3": "binary_crossentropy",
                         "alpha_4": "binary_crossentropy", "alpha_5": "binary_crossentropy"}
            self.metrics = [categorical_accuracy, binary_accuracy, binary_accuracy, binary_accuracy, binary_accuracy,
                            binary_accuracy]
            self.model.compile(metrics=self.metrics, optimizer=tf.optimizers.Adam(learning_rate=0.02), loss=self.loss)

        else:

            print("Only stance!!!!")
            outputs = [stance_probs]
            self.model = Model(inputs=inputs, outputs=outputs)
            self.loss = "categorical_crossentropy"

            self.model.compile(metrics=[categorical_accuracy], optimizer=tf.optimizers.Adam(learning_rate=0.02),
                               loss=self.loss)

            self.model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.02),
                               metrics=['accuracy'])

        self.model.summary()

    def train(self, x_train, x_test, y_train, y_test, train_gold_ev, test_gold_ev, number_sents=5):
        model_metrics = NewMetrics(x_train, y_train, x_test, y_test, train_gold_ev, test_gold_ev, only_stance=self.only_stance,
                 reversed_stance_dict=self.reversed_stance_dict, number_sents=number_sents)
        early_stopper = EarlyStopping(monitor='val_loss', patience=10, mode="min")

        history = self.model.fit(x_train, y_train, \
                                 batch_size=self.batch_size, \
                                 epochs=self.epochs, \
                                 callbacks=[early_stopper, model_metrics], \
                                 validation_data=(x_test, y_test), \
                                 )
