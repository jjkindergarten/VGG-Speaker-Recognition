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
parser.add_argument('--resume', default='../model/gvlad_softmax/2021-02-05_resnet34s_bs16_adam_'
                                        'lr0.001_vlad8_ghost2_bdim512_ohemlevel0/weights-18-0.968.h5', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--data_path', default='../../../data/audio_data_OTR', type=str)
parser.add_argument('--multiprocess', default=4, type=int)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)

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

    vallist, vallb = toolkits.get_hike_datalist(args, path='../meta/hike_val.json')
    # vallist = vallist[:5]
    # vallb = vallb[:5]

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
    params = {'dim': (257, None, 1),
              'mp_pooler': toolkits.set_mp(processes=args.multiprocess),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 2,
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
                                          params['nfft'], params['spec_len'])) for ID in vallist]
    val_data = np.expand_dims(np.array([p.get() for p in val_data]), -1)

    v = network_eval.predict(val_data)
    print(v.shape)
    v = ((v<0.5)*1)[:,0]
    acc = sum(v==vallb)/len(vallb)
    print(v)
    print(vallb)
    print(acc)


if __name__ == '__main__':
    main()