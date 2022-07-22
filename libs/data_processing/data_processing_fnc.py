import os
import pandas as pd

if __name__ == "__main__":
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_bodies = pd.read_csv(path+"/dataset/fnc/competition_test_bodies.csv")
    train_stances = pd.read_csv(path+"/dataset/fnc/competition_test_stances.csv")


    body_id = train_bodies['Body ID']
    article_body = train_bodies['articleBody']
    stances = train_stances['Stance']
    target = train_stances['Headline']
    cor_body_id = train_stances['Body ID']

    bodies_dict = {}
    for i in range(len(body_id)):
        bodies_dict[body_id[i]] = article_body[i]

    article_list = []

    for i in cor_body_id:
        article_list.append(bodies_dict[i])

    # turn stance to number
    stance_num = list(set(stances))
    stance_dict = {}
    for i in range(len(stance_num)):
        stance_dict[stance_num[i]] = i

    label_list = []
    for i in range(len(stances)):
        label_list.append(stance_dict[stances[i]])

    # save
    data_dict = {'article': article_list,
                 'title': target,
                 'target': target,
                 'stance': stances,
                 'label': label_list,
                 }

    data_csv = pd.DataFrame.from_dict(data_dict)
    data_csv.to_csv(path+'/dataset/fnc/test.csv')
