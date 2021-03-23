import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from toolkits import assign_category

model_config_file = '../../model_config/vgg_speaker_config.json'
with open(model_config_file, 'r') as f:
    model_config = json.load(f)
    f.close()

score_rule_file = '../../model_config/classification_rule.json'
with open(score_rule_file, 'r') as f:
    score_rule = json.load(f)
    f.close()

def assign_score(low_prob, high_prob):
    if (low_prob < 0.4) and (high_prob > 0.6):
        # high category (7, 8, 9)
        if high_prob < 0.8:
            return 7
        elif high_prob < 0.9:
            return 8
        else:
            return 9
    if (low_prob > 0.6) and (high_prob < 0.4):
        # low category (2, 3, 4)
        if low_prob < 0.8:
            return 4
        elif low_prob < 0.9:
            return 3
        else:
            return 2
    if (low_prob < 0.6) and (high_prob < 0.6):
        # medium category (5, 6)
        if low_prob < high_prob:
            return 6
        elif low_prob > high_prob:
            return 5
    print('{} and {} is not solved yet'.format(low_prob, high_prob))
    return 5



def calculate_acc(content_table, score_rule):
    for cate in score_rule:
        max_threshold = score_rule[cate]['max']
        min_threshold = score_rule[cate]['min']
        content_select = content_table[(content_table['score_true'] > min_threshold) &
                                     (content_table['score_true'] < max_threshold)][['score_true', 'score_predict']].copy()
        content_select = ((content_select > min_threshold) & (content_select < max_threshold)) * 1
        acc = sum(content_select['score_true'] == content_select['score_predict']) / len(content_select)
        print('accuracy of category {} is: {}'.format(cate, acc))
    result = content_table.apply(lambda x: assign_category(score_rule, x))
    print('confusion matrix: ')
    print(confusion_matrix(result['score_true'].values, result['score_predict'].values))

def calculate_mse(content_table, score_rule):
    for cate in score_rule:
        max_threshold = score_rule[cate]['max']
        min_threshold = score_rule[cate]['min']
        content_select = content_table[(content_table['score_true'] > min_threshold) &
                                     (content_table['score_true'] < max_threshold)][['score_true', 'score_predict']].copy()
        mse = np.square(np.subtract(content_select['score_predict'].values, content_select['score_true'].values)).mean()
        print('mse of category {} is: {}'.format(cate, mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table1_path', default='../../../../data/spec_data_OTR2', type=str)
    parser.add_argument('--table2_path', default='', type=str)
    # parser.add_argument('--meta_table_path', default='', type=str)
    args = parser.parse_args()



    if model_config['loss'] != 'mse':
        assert args.table2_path != ''
        high_table = pd.read_csv(args.table1_path)
        low_table = pd.read_csv(args.table2_path)
        high_table.rename({'prob_0':'high_prob_0', 'prob_1':'high_prob_1', 'true_label':'high_true_label'})
        low_table.rename({'prob_0':'low_prob_0', 'prob_1':'low_prob_1', 'true_label':'low_true_label'})
        content_table = high_table.merge(low_table, how='inner', on='content')
        content_table['score_predict'] = ''
        for i in range(len(content_table)):
            low_prob = content_table.loc[i, 'low_prob_0']
            high_prob = content_table.loc[i, 'high_prob_1']
            content_table.loc[i, 'score_predict'] = assign_score(low_prob, high_prob)
    else:
        content_table = pd.read_csv(args.table1_path)

    calculate_acc(content_table, score_rule)
    calculate_mse(content_table, score_rule)



