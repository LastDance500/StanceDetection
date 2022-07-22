import sys, os
import numpy
import random as rd
import json
from nltk import word_tokenize, sent_tokenize

STANCE_DICT={"support": 0, "refute": 1, "comment": 2, "unrelated": 3}


def dump_json(content, filename):
	with open(filename, "w") as fout:
		json.dump(content, fout, indent=4)


def load_json(filename):
	with open(filename, "r") as fin:
		return json.load(fin)


def get_target(operation):
	dict_operations = {"AET": "Aetna",\
						"CI": "Cigna",\
						"CVS": "CVS",\
						"ANTM": "Anthem",\
						"HUM": "Humana",\
						"ESRX": "Express Script"}
	buyer, target = operation.split("_")[0], operation.split("_")[1]
	claim = dict_operations[buyer] + " (" + buyer + ") will merge with " + dict_operations[target] + " (" + target + ")."
	return claim


def retrieve_sentences(all_sentences):
	pass


def split_train_test(data, stances, op, get_gold_sentences=True, number_sents=5):
	train, test = {"stances": [], "targets": [], "sentences": [], "evidence_labels": [], "gold_ev": []},\
					{"stances": [], "targets": [], "sentences": [], "evidence_labels": [], "gold_ev": []}

	for file_id, file_data in data.items():
		# load target
		file_operation = "_".join(file_id.split("_")[:2])
		target = get_target(file_operation)
		# load sentences
		sentences = []
		evidence_indices = []
		gold_ev = []

		# retrieve sentences
		if get_gold_sentences:
			if len(file_data["evidence_title_indices"]) > 0:
				sentences.append(file_data["title"])
				evidence_indices.append(1)
			if len(file_data["evidences_sents"]) > 0:
				for evidence_sent_index in set(file_data["evidences_sents"]):
					evidence_sent = file_data["sentences"][evidence_sent_index]
					if len(sentences) < 5:
						sentences.append(evidence_sent)
						evidence_indices.append(1)
		else:
			# title
			if len(file_data["evidence_title_indices"]) > 0:
				evidence_indices.append(1)
			else:
				evidence_indices.append(0)
			sentences.append(file_data["title"])
			# sentences
			sentences_from_body = file_data["sentences"][:(number_sents-1)]
			for sent_i, sent in enumerate(sentences_from_body):
				sentences.append(sent)
				if sent_i in file_data["evidences_sents"]:
					evidence_indices.append(1)
				else:
					evidence_indices.append(0)
			assert len(evidence_indices) == len(sentences)

		# retrieve gold indices
		if len(file_data["evidence_title_indices"]) > 0:
			gold_ev.append(0)
			if len(file_data["evidences_sents"]) > 0:
				all_ev = set(file_data["evidences_sents"])
				for e in all_ev:
					gold_ev.append(e+1)
		
		if len(evidence_indices) < number_sents:
			add_zeros = [0] * (number_sents - len(evidence_indices))
			empty_sents = [""] * (number_sents - len(evidence_indices))
			evidence_indices = evidence_indices + add_zeros
			sentences = sentences + empty_sents

		assert len(evidence_indices) == number_sents
		assert len(evidence_indices) == len(sentences)

		# get stance
		stance = stances[file_id]["gold_stance"].strip()

		# append to train or test
		if not (stance.startswith("unselected") or stance.startswith("unrelated")):
			if file_id.startswith(op):
				test["stances"].append(stance)
				test["targets"].append(target)
				test["sentences"].append(sentences)
				test["evidence_labels"].append(evidence_indices)
				test["gold_ev"].append(gold_ev)

			else:
				train["stances"].append(stance)
				train["targets"].append(target)
				train["sentences"].append(sentences)
				train["evidence_labels"].append(evidence_indices)
				train["gold_ev"].append(gold_ev)
	
	# shuffle
	#permutation = numpy.random.permutation(len(train["stances"]))
	
	fixed_seed = rd.random()
	rd.Random(fixed_seed).shuffle(train["stances"])
	rd.Random(fixed_seed).shuffle(train["targets"])
	rd.Random(fixed_seed).shuffle(train["sentences"])
	rd.Random(fixed_seed).shuffle(train["evidence_labels"])
	rd.Random(fixed_seed).shuffle(train["gold_ev"])
	
	# convert stance categorical labels to one-shot
	return train, test


def sent_to_indices(sent, indices_dict, maxlen=25):
	indices = []
	for w in word_tokenize(sent):
		w = w.lower()
		if w in indices_dict:
			indices.append(indices_dict[w])
		else:
			indices.append(0)

	# pad
	if len(indices) < maxlen:
		add_zeros = [0] * (maxlen - len(indices))
		indices = indices + add_zeros
	indices = indices[:maxlen]
	
	return indices


def preprocess_bert(all_data, data_bert, bert_matrix, stances, target_embs, op=None, only_stance=False, number_sents=5, no_classes=4):
	x_train, x_test = [[], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], []]
	train_gold_ev, test_gold_ev = [], []

	dict_operations = {"AET_HUM": 0,\
						"ANTM_CI": 1,\
						"CI_ESRX": 2,\
						"CVS_AET": 3}

	if only_stance:
		y_train, y_test = [], []
	else:
		y_train, y_test = [[] for i in range(number_sents+1)], [[] for i in range(number_sents+1)]

	# get data
	for file_id, file_content in data_bert.items():
		# get stance
		stance = stances[file_id]["gold_stance"].strip()

		# get target
		file_operation = "_".join(file_id.split("_")[:2])
		target_emb = target_embs[dict_operations[file_operation]]

		if only_stance:
			# get gold sentences
			s, e = file_content["sentence_indices"][0], file_content["sentence_indices"][1]
		else:
			# get all sentences
			s, e = file_content["sentence_indices"][0], file_content["sentence_indices"][1]
		
		s1, s2, s3, s4, s5 = bert_matrix[s], bert_matrix[s+1], bert_matrix[s+2], bert_matrix[s+3], bert_matrix[s+4]

		text = all_data[file_id]['title']
		for t in all_data[file_id]['sentences']:
			text += t

		target = all_data[file_id]['target']

		# add to train or test
		if not stance.startswith("unselected") and not stance.startswith("unrelated"):
			if file_id.startswith(op):
				test_gold_ev.append(file_content["gold_evidences"])
				x_test[0].append(target_emb)
				x_test[1].append(s1)
				x_test[2].append(s2)
				x_test[3].append(s3)
				x_test[4].append(s4)
				x_test[5].append(s5)
				x_test[6].append(text)
				x_test[7].append(target)

				if not only_stance:
					sample_evidences = file_content["evidence_indices"]
					y_test[0].append(STANCE_DICT[stance])
					for i in range(1, number_sents+1):
						y_test[i].append(sample_evidences[i-1])
				else:
					y_test.append(STANCE_DICT[stance])
			else:
				train_gold_ev.append(file_content["gold_evidences"])
				x_train[0].append(target_emb)
				x_train[1].append(s1)
				x_train[2].append(s2)
				x_train[3].append(s3)
				x_train[4].append(s4)
				x_train[5].append(s5)
				x_train[6].append(text)
				x_train[7].append(target)
				if not only_stance:
					sample_evidences = file_content["evidence_indices"]
					y_train[0].append(STANCE_DICT[stance])
					for i in range(1, number_sents+1):
						y_train[i].append(sample_evidences[i-1])
				else:
					y_train.append(STANCE_DICT[stance])

	# shuffle train!
	fixed_seed = rd.random()
	rd.Random(fixed_seed).shuffle(x_train[0])
	rd.Random(fixed_seed).shuffle(x_train[1])
	rd.Random(fixed_seed).shuffle(x_train[2])
	rd.Random(fixed_seed).shuffle(x_train[3])
	rd.Random(fixed_seed).shuffle(x_train[4])
	rd.Random(fixed_seed).shuffle(x_train[5])
	rd.Random(fixed_seed).shuffle(x_train[6])
	rd.Random(fixed_seed).shuffle(x_train[7])

	if not only_stance:
		rd.Random(fixed_seed).shuffle(y_train[0])
		rd.Random(fixed_seed).shuffle(y_train[1])
		rd.Random(fixed_seed).shuffle(y_train[2])
		rd.Random(fixed_seed).shuffle(y_train[3])
		rd.Random(fixed_seed).shuffle(y_train[4])
		rd.Random(fixed_seed).shuffle(y_train[5])
		rd.Random(fixed_seed).shuffle(y_train[6])
		rd.Random(fixed_seed).shuffle(y_train[7])
	else:
		rd.Random(fixed_seed).shuffle(y_train)

	# convert everything
	for i in range(6):
		x_train[i] = numpy.array(x_train[i])
		x_test[i] = numpy.array(x_test[i])

	# if not only_stance:
	# 	y_train[0], y_test[0] = to_categorical(y_train[0], no_classes), to_categorical(y_test[0], no_classes)
	# 	for i in range(1, number_sents+1):
	# 		y_train[i], y_test[i] = numpy.array(y_train[i]), numpy.array(y_test[i])
	# else:
	# 	y_train, y_test = to_categorical(y_train, no_classes), to_categorical(y_test, no_classes)

	return x_train, y_train, x_test, y_test, train_gold_ev, test_gold_ev


if __name__ == '__main__':
	# define experiment names
	exp_name = 'a'
	test_operation = 'AET_HUM'
	encoder = 'BERT'

	# read data
	data = load_json("final_merged_annotations_correct.json")
	stances = load_json("final_merged_annotations_correct.json")
	train, test = split_train_test(data, stances, test_operation, get_gold_sentences=False)

	only_stance = False
	try:
		training_setting = 'True'
		if training_setting == "True":
			only_stance = True
	except:
		pass

	if encoder == "BERT":
		data_bert = load_json("bert_converter.json")
		bert_matrix = numpy.load("bert_embeddings.npy")
		x_train, y_train, x_test, y_test, train_gold_ev, test_gold_ev = preprocess_bert(data_bert, bert_matrix, stances, op=test_operation, only_stance=only_stance)

		bilstm_dense = 68
		use_dense_size = 68*2
		BERT_dense = True
		USE_dense = True

	model = WingOsModel(experiment_name=exp_name,\
						only_stance=only_stance,\
						encoder=encoder,\
						bilstm_dense=bilstm_dense,\
						USE_dense=USE_dense,
						BERT_dense=BERT_dense,\
						use_dense_size=use_dense_size)
	model.build_model()
	model.train(x_train, x_test, y_train, y_test, train_gold_ev, test_gold_ev, number_sents=5)
