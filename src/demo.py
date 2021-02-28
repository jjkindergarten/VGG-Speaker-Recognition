from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import librosa
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
parser.add_argument('--data_path', default='../../../../data/spec_data_OTR', type=str)
parser.add_argument('--save_dir', default='', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax', 'regression'], type=str)

global args
args = parser.parse_args()

def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model

    params = {'dim': (257, None, 1),
              'n_fft': 512,
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

    wav, sr_ret = librosa.load(args.data_path, sr=params['sampling_rate'])
    linear_spect = ut.lin_spectogram_from_wav(wav, params['hop_length'], params['win_length'], params['n_fft'])

    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    spec_mag = mag_T
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag*(10**5), 0, keepdims=True)/(10**5)
    spec_mag = (spec_mag - mu) / (std + 1e-3)
    spec_mag = np.expand_dims(spec_mag, (0, -1))
    v = network_eval.predict(spec_mag)
    print('the predicted score is: {}'.format(v))

if __name__ == '__main__':
    main()