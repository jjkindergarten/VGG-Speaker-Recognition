import os
import json
import numpy as np

import src.model as model
import src.utils as ut

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV, RFE
from sklearn.svm import SVC

data_dir = '../../data/audio_data_OTR'

with open(os.path.join('../../data', 'hike_meta_data_combine.json'), 'r') as f:
    meta_list = json.load(f)

args = {
    'net': 'resnet34s',
    'loss': 'softmax',
    'vlad_cluster': 8,
    'ghost_cluster': 2,
    'bottleneck_dim': 512,
    'aggregation_mode': 'gvlad'
}

params = {'dim': (257, None, 1),
          'nfft': 512,
          'spec_len': 250,
          'win_length': 400,
          'hop_length': 160,
          'n_classes': 5994,
          'sampling_rate': 16000,
          'normalize': True,
          }

meta_list1 = meta_list[0]
audio_set = [(i[2], i[3], i[1]) for i in meta_list1]

network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                            num_class=params['n_classes'],
                                            mode='eval', args=args)

network_eval.load_weights('./model/weights.h5', by_name=True)

feats, scores, labels = [], [], []
for id, audio_name, cate in audio_set:
    audio_path = os.path.join(data_dir, audio_name)
    specs = ut.load_data(audio_path, win_length=params['win_length'], sr=params['sampling_rate'],
                         hop_length=params['hop_length'], n_fft=params['nfft'],
                         spec_len=params['spec_len'], mode='eval')

    specs = np.expand_dims(np.expand_dims(specs, 0), -1)

    v = network_eval.predict(specs)
    feats.append(v[0].tolist())

with open(os.path.join('../../data', 'vgg_speaker_feature_hike_combine.json'), 'w') as f:
    json.dump(feats, f)

# meta_list2 = meta_list[1]
# id_list = [filename[2] for filename in meta_list2]
# category_list = [category[1] for category in meta_list2]
#
# id_train, id_test, y_train, y_test = train_test_split(id_list, category_list, test_size=0.2, random_state=32)
#
# train_meta_list = [(i[3], i[1]) for i in meta_list1 if i[2] in id_train]
# test_meta_list = [(i[3], i[1]) for i in meta_list1 if i[2] in id_test]
#
# audio_set2 = [i[1] for i in audio_set]
#
# train_X = np.zeros(shape=(len(train_meta_list), 512))
# train_y = np.zeros(shape=(len(train_meta_list, )), dtype=object)
# for i, (audio_path, score) in enumerate(train_meta_list):
#     audio_index = audio_set2.index(audio_path)
#     train_X[i] = feats[audio_index]
#     train_y[i] = score
#
# test_X = np.zeros(shape=(len(test_meta_list), 512))
# test_y = np.zeros(shape=(len(test_meta_list, )), dtype=object)
# for i, (audio_path, score) in enumerate(test_meta_list):
#     audio_index = audio_set2.index(audio_path)
#     test_X[i] = feats[audio_index]
#     test_y[i] = score
#
# clf = LogisticRegression(penalty='l1', random_state=3, solver='saga').fit(train_X, train_y)
# test_res = np.array([test_y, clf.predict(test_X)]).T
# print('score of logistic regression: ', clf.score(test_X, test_y))
#
# rf_classifier = RandomForestClassifier(n_estimators=1200, min_samples_split=25)
# rf_classifier.fit(train_X, train_y)
#
#
# estimator = SVC(kernel="linear")
# selector = RFE(estimator, step=0.1, n_features_to_select=10)
# selector = selector.fit(train_X, train_y)
# print('num of feature remains: ', selector.n_features_)
#
# train_audio_list_selected = train_X[:, selector.support_]
# test_audio_list_selected = test_X[:, selector.support_]
# clf = SVC(kernel="rbf").fit(train_audio_list_selected, train_y)
# test_res = np.array([test_y, clf.predict(test_audio_list_selected)]).T
# print('score of svc after RFE: ', clf.score(test_audio_list_selected, test_y))