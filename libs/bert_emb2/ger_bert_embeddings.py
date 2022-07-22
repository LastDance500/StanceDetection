import sys, os
import json
import numpy


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')


def load_json(filename):
	with open(filename, "r") as fin:
		return json.load(fin)


def dump_json(content, filename):
	with open(filename, "w") as fout:
		json.dump(content, fout, indent=4)


def get_target_embeddings():
	dict_operations = {"AET": "Aetna",\
						"CI": "Cigna",\
						"CVS": "CVS",\
						"ANTM": "Anthem",\
						"HUM": "Humana",\
						"ESRX": "Express Script"}

	targets = ["Aetna (AET) will merge with Humana (HUM).",\
				"Anthem (ANTM) will merge with Cigna (CI).",\
				"Cigna (CI) will merge with Express Script (ESRX).",\
				"CVS (CVS) will merge with Aetna (AET)."]

	dict_operations = {"AET_HUM": 0,\
						"ANTM_CI": 1,\
						"CI_ESRX": 2,\
						"CVS_AET": 3}

	target_embeddings = model.encode(targets)
	target_embeddings = numpy.array(target_embeddings)
	numpy.save("bert_target_embeddings.npy", target_embeddings)


def get_sentence_embeddings(data, number_sents=5):
	dict_sentences = {}
	all_sentence_embeddings = "Empty"
	
	for file_id, file_data in data.items():
		print("file_id: ", file_id)
		
		gold_sentences, sentences, evidence_indices, gold_evidences = [], [], [], []

		# get GOLD sentences
		if len(file_data["evidence_title_indices"]) > 0:
			gold_sentences.append(file_data["title"])
		if len(file_data["evidences_sents"]) > 0:
			for evidence_sent_index in set(file_data["evidences_sents"]):
				evidence_sent = file_data["sentences"][evidence_sent_index]
				if len(sentences) < 5:
					gold_sentences.append(evidence_sent)
		
		# get NOT GOLD sentences
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
			gold_evidences.append(0)
			if len(file_data["evidences_sents"]) > 0:
				all_ev = set(file_data["evidences_sents"])
				for e in all_ev:
					gold_evidences.append(e+1)
		
		if len(gold_sentences) == 0:
			gold_sentences.append(file_data["title"])
			for sent_i, sent in enumerate(sentences_from_body):
				gold_sentences.append(sent)
				

		if len(evidence_indices) < number_sents:
			add_zeros = [0] * (number_sents - len(evidence_indices))
			empty_sents = [""] * (number_sents - len(evidence_indices))
			evidence_indices = evidence_indices + add_zeros
			sentences = sentences + empty_sents
			if len(gold_sentences) > 0:
				empty_sents_gold = [""] * (number_sents - len(gold_sentences))
				gold_sentences = gold_sentences + empty_sents_gold

		# convert sentences
		gold_sents_embeddings = model.encode(gold_sentences)
		sent_embeddings = model.encode(sentences)

		gold_sents_embeddings = numpy.array(gold_sents_embeddings)
		sent_embeddings = numpy.array(sent_embeddings)
		print("gold_sents_embeddings", gold_sents_embeddings.shape)
		print(gold_evidences, gold_sentences)
		print("sent_embeddings", sent_embeddings.shape)
		if all_sentence_embeddings == "Empty":
			start_gold = 0
			all_sentence_embeddings = gold_sents_embeddings
			start_sents = len(all_sentence_embeddings)
			print("all_sentence_embeddings", all_sentence_embeddings.shape)
			print("sent_embeddings", sent_embeddings.shape)
			all_sentence_embeddings = numpy.concatenate([all_sentence_embeddings, sent_embeddings])
		else:
			start_gold = len(all_sentence_embeddings)
			print("all_sentence_embeddings", all_sentence_embeddings.shape)
			print("sent_embeddings", sent_embeddings.shape)
			all_sentence_embeddings = numpy.concatenate([all_sentence_embeddings, gold_sents_embeddings])
			start_sents = len(all_sentence_embeddings)
			print("all_sentence_embeddings", all_sentence_embeddings.shape)
			all_sentence_embeddings = numpy.concatenate([all_sentence_embeddings, sent_embeddings])

		dict_sentences[file_id] = {"gold_sentences_indices": [start_gold, len(gold_sents_embeddings)],\
									"sentence_indices": [start_sents, len(sent_embeddings)],\
									"evidence_indices": evidence_indices,\
									"gold_evidences": gold_evidences}

	# at the end, save embedding matrix and bert dict
	numpy.save("bert_embeddings.npy", all_sentence_embeddings)
	dump_json(dict_sentences, "bert_converter.json")


if __name__ == '__main__':
	data = load_json("final_merged_annotations_correct.json")

	get_target_embeddings()
	get_sentence_embeddings(data)




