import json
import pickle
import pandas as pd
from tqdm import tqdm
import re
from stanfordcorenlp import StanfordCoreNLP

STANCE_DICT = {"support": 0, "refute": 1, "comment": 2, "unrelated": 3}


def load_json(filename):
    with open(filename, "r") as fin:
        return json.load(fin)


if __name__ == '__main__':
    nlp = StanfordCoreNLP(r'G:\github\stanford-corenlp-4.4.0')
    infile_train = open('data/data_train.pkl', 'rb')
    data_train = pickle.load(infile_train)

    data = load_json("data/final_merged_annotations_correct.json")

    news_list = []
    stance_list = []
    target_list = []
    title_list = []
    tokens_list = []
    no_of_tokens_list = []
    dependency_list = []
    dependency_tree_list = []
    dependency_edge_list = []
    target_token_list = []
    target_index_list = []
    target_name_list = []
    overall_graph_tokens = []
    overall_graph_edges = []
    overall_graph_names = []
    overall_graph_root = []
    for k in tqdm(data.keys()):
        sample = data.get(k)

        try:
            stance = STANCE_DICT[sample['gold_stance']]
            stance_list.append(stance)
        except Exception as e:
            print(e)
            continue

        target = sample['target'].lower()
        target_list.append(target)

        target_token = nlp.word_tokenize(target)
        target_token_list.append(target_token)

        target_name = re.sub(r"\d+", '', k)[:-1]

        target_name_list.append(target_name)

        target_len = len(target_token)
        target_index = [i for i in range(target_len)]
        target_index_list.append(target_index)

        sample_sentence = sample['sentences']
        # if len(sample_sentence) >= 5:
        #     sentence = [target[:-1] + " " + sample_sentence[i].lower() for i in range(5)]
        # else:
        #     sentence = [target[:-1] + " " + sample_sentence[i].lower() for i in range(len(sample_sentence))]
        #     for i in range(5 - len(sample_sentence)):
        #         sentence.append(target[:-1] + ' ')

        sentence = [sample_sentence[i].lower() for i in range(len(sample_sentence))]

        news_list.append(sentence)

        title = sample['title'].lower()
        title_list.append(title)

        sentence = [target] + [title] + sentence

        whole_sentence = target + ' ' + title + ' '.join(sentence)

        dependency_tree_temp_list = []
        dependency_temp_list = []
        dependency_edge_temp_list = []
        for j in range(len(sentence)):
            sen = sentence[j]
            dependency_tree = nlp.dependency_parse(sen)
            dependency_tree_temp_list.append(dependency_tree)

            dependency = [a for (a, _, _) in dependency_tree]
            dependency_temp_list.append(dependency)

            dependency_edge = [[b, c] for (_, b, c) in dependency_tree]
            dependency_edge_temp_list.append(dependency_edge)


        # process the graph list to a large graph
        len_flag = 0
        graph_tokens = []
        graph_names = []
        graph_edges = []
        for m in range(len(dependency_tree_temp_list)):
            sen_graph = dependency_tree_temp_list[m]
            s = sentence[m]

            tokens = nlp.word_tokenize(sentence[m])
            tokens = ['ROOT'] + tokens
            graph_tokens += tokens
            for i in range(len(sen_graph)):
                node = sen_graph[i]
                name = node[0]

                graph_names.append(name)

                edge = [len_flag + node[1], len_flag + node[2]]
                graph_edges.append(edge)

            len_flag += len(sen_graph)

        root_index = []
        for i in range(len(graph_names)):
            n = graph_names[i]
            if n == 'ROOT':
                root_index.append(i)

        root_edge = []
        for i in range(len(root_index)-1):
            if i == 0:
                for j in range(len(root_index)-2):
                    root_edge.append([root_index[0], root_index[j]])
            else:
                root_edge.append([root_index[i], root_index[i+1]])

        overall_graph_root.append(root_index)
        overall_graph_names.append(graph_names)
        overall_graph_tokens.append(graph_tokens)
        overall_graph_edges.append(graph_edges+root_edge)


        dependency_tree_list.append(dependency_tree_temp_list)
        dependency_list.append(dependency_temp_list)
        dependency_edge_list.append(dependency_edge_temp_list)

        tokens = [nlp.word_tokenize(s) for s in sentence]
        tokens_list.append(tokens)

        no_of_tokens = [len(t) for t in tokens]
        no_of_tokens_list.append(no_of_tokens)

    stander_dict = {'news': news_list, 'title': title_list, 'stance': stance_list,
                    'dependency_tree': dependency_tree_list,
                    'dependency': dependency_list, 'dependency_edge': dependency_edge_list,
                    'overall_graph_tokens': overall_graph_tokens, 'overall_graph_edges': overall_graph_edges,
                    'overall_graph_names': overall_graph_names, 'target_name': target_name_list,
                    'target_index': target_index_list, 'target_token': target_token_list, 'tokens': tokens_list,
                    'no of tokens': no_of_tokens_list}

    pd.DataFrame(stander_dict).to_csv('data/stander_new3.csv', index=False)

    print(1)
