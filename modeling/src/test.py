from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import json
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import confusion_matrix

sys.path.append('../tool')
import toolkits
import utils as ut

import pdb
# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='../../model_weight/regression_weight.h5', type=str)
# parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--test_data_path', default='../../../D2_data/audio_data_OTR', type=str, nargs='+')
parser.add_argument('--multiprocess', default=4, type=int)
# set up network configuration.
# parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
# parser.add_argument('--ghost_cluster', default=2, type=int)
# parser.add_argument('--vlad_cluster', default=8, type=int)
# parser.add_argument('--bottleneck_dim', default=512, type=int)
# parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
# parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax', 'regression'], type=str)
parser.add_argument('--test_meta_data_path', default='../../../D2_data/meta2', type=str,  nargs='*')
parser.add_argument('--save_dir', default='', type=str)
# parser.add_argument('--seed', default=2, type=int, help='seed for which dataset to use')
parser.add_argument('--data_format', default='wav', choices=['wav', 'npy'], type=str)
parser.add_argument('--audio_length', default=20, type=float)
# parser.add_argument('--category', default='high_low', type=str)

global args
args = parser.parse_args()

global model_config
model_config_file = '../../model_config/vgg_speaker_config.json'
with open(model_config_file, 'r') as f:
    model_config = json.load(f)
    f.close()

global score_rule
score_rule_file = '../../model_config/classification_rule.json'
with open(score_rule_file, 'r') as f:
    score_rule = json.load(f)
    f.close()

def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model
    # ==================================
    #       Get Train/Val.
    # ==================================
    vallist, vallb = toolkits.get_hike_datalist(meta_paths=args.test_meta_data_path, data_paths=args.test_data_path, mode=model_config['loss'])
    _, valscore = toolkits.get_hike_datalist(meta_paths=args.test_meta_data_path, data_paths=args.test_data_path, mode='mse')

    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.
    num_class = len(score_rule)
    input_length = int(args.audio_length * 25)
    params = {'dim': (513, None, 1),
              'mp_pooler': toolkits.set_mp(processes=args.multiprocess),
              'nfft': 1024,
              'spec_len': input_length,
              'win_length': 1024,
              'hop_length': 640,
              'n_classes': num_class,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=model_config)
    # ==> load pre-trained model ???
    if args.resume:
        # ==> get real_model from arguments input,
        # load the model if the imag_model == real_model.
        if os.path.isfile(args.resume):
            network_eval.load_weights(os.path.join(args.resume), by_name=True, skip_mismatch=True)
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            raise IOError("==> no checkpoint found at '{}'".format(args.resume))
    else:
        raise IOError('==> please type in the model to load')

    print('==> start testing.')

    v = []
    for ID in vallist:
        val_data = ut.load_data(ID, params['win_length'], params['sampling_rate'], params['hop_length'],params['nfft'],
                     params['spec_len'], 'test', args.data_format)
        info = network_eval.predict(np.expand_dims(val_data, (0, -1)))
        v += info.tolist()
    v = np.array(v)

    print('val data shape {}'.format(v.shape))
    if model_config['loss'] == 'mse':
        v = v.T[0] * 10 + 5
        vallb = vallb * 10 + 5
        metric = np.square(np.subtract(v, vallb)).mean()
        print('mse: ', metric)
        v_test = np.vstack([v, vallb]).astype('float').T
        df = np.hstack([vallist.reshape(-1, 1), v_test])
        df = pd.DataFrame(data=df, columns=['content', 'score_predict', 'score_true'])
    else:
        v_predict = ((v<0.5)*1)[:,0]
        metric = sum(v_predict==vallb)/len(vallb)
        print('confusion matrix: ', confusion_matrix(vallb, v_predict))
        print('accuracy ', metric)
        v_test = np.hstack([v, vallb.reshape(-1, 1), valscore]).astype('float')
        df = np.hstack([vallist.reshape(-1, 1), v_test])
        df = pd.DataFrame(data=df, columns=['content', 'prob_0', 'prob_1', 'true_label', 'score_true'])

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    df.to_csv(os.path.join(args.save_dir, '{}_{}_{}.csv'.format(date, model_config['loss'], metric)))

if __name__ == '__main__':
    main()