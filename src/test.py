from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np

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
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--data_path', default='../../../D2_data/audio_data_OTR', type=str)
parser.add_argument('--multiprocess', default=4, type=int)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax', 'regression'], type=str)
parser.add_argument('--meta_data_path', default='../../../D2_data/meta2', type=str)
parser.add_argument('--seed', default=2, type=int, help='seed for which dataset to use')
parser.add_argument('--data_format', default='wav', choices=['wav', 'npy'], type=str)
parser.add_argument('--audio_length', default=2.5, type=float)
parser.add_argument('--category', default='high_low', type=str)

global args
args = parser.parse_args()

def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model
    # ==================================
    #       Get Train/Val.
    # ==================================
    # print('==> calculating test({}) data lists...'.format(args.test_type))
    category = args.category.split('_')
    if args.loss != 'regression':
        testlist, testlb = toolkits.get_hike_datalist(category, meta_path=os.path.join(args.meta_data_path,
                                                                            'hike_test_{}.json'.format(args.seed)),
                                                    data_path=args.data_path)
        vallist, vallb = toolkits.get_hike_datalist(category, meta_path=os.path.join(args.meta_data_path,
                                                                            'hike_val_{}.json'.format(args.seed)),
                                                    data_path=args.data_path)

    else:
        testlist, testlb = toolkits.get_hike_datalist2(category, meta_path=os.path.join(args.meta_data_path,
                                                                            'hike_test_{}.json'.format(args.seed)),
                                                    data_path=args.data_path)
        vallist, vallb = toolkits.get_hike_datalist2(category, meta_path=os.path.join(args.meta_data_path,
                                                                            'hike_val_{}.json'.format(args.seed)),
                                                    data_path=args.data_path)

    vallist = np.concatenate([vallist, testlist])
    vallb = np.concatenate([vallb, testlb])
    # vallist = vallist[:10]
    # vallb = vallb[:10]

    #
    # verify_list = np.loadtxt('../meta/voxceleb1_veri_test_extended.txt', str)
    #
    # verify_lb = np.array([int(i[0]) for i in verify_list])
    # list1 = np.array([os.path.join(args.data_path, i[1]) for i in verify_list])
    # list2 = np.array([os.path.join(args.data_path, i[2]) for i in verify_list])
    #
    # total_list = np.concatenate((list1, list2))
    # unique_list = np.unique(total_list)

    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.
    num_class = len(set(vallb))
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
                                                mode='eval', args=args)

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

    val_data = [params['mp_pooler'].apply_async(ut.load_data,
                                    args=(ID, params['win_length'], params['sampling_rate'], params['hop_length'],
                                          params['nfft'], params['spec_len'], 'test', args.data_format)) for ID in vallist]
    val_data = np.expand_dims(np.array([p.get() for p in val_data]), -1)
    # for ID in vallist:
    #     val_data = ut.load_data(ID, params['win_length'], params['sampling_rate'], params['hop_length'],params['nfft'],
    #                  params['spec_len'], 'train', args.data_format)

    print(val_data.shape)
    v = network_eval.predict(val_data)
    if args.loss == 'regression':
        v = v.T[0] * 10 + 5
        vallb = vallb * 10 + 5
        print(v.shape)
        print(v)
        print(vallb)
        print('mse: ', np.square(np.subtract(v, vallb)).mean())
        v_test = np.vstack([v, vallb]).astype('float').T
        toolkits.get_content_score(args.meta_data_path, v_test, args)
    else:
        v = ((v<0.5)*1)[:,0]
        acc = sum(v==vallb)/len(vallb)
        print(v)
        print(vallb)
        print(acc)

if __name__ == '__main__':
    main()