from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import keras
import wandb
from wandb.keras import WandbCallback
from collections import  Counter
import json
import numpy as np
import utils as ut


np.seterr(all='raise')

sys.path.append('../tool')
import toolkits

# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='', type=str)
# parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--train_data_path', default='../../D2_data/audio_data_OTR', type=str, action='append', nargs='*')
parser.add_argument('--train_meta_data_path', default='../../D2_data/meta2', type=str, action='append', nargs='*')
parser.add_argument('--val_data_path', default='../../D2_data/audio_data_OTR', type=str, action='append', nargs='*')
parser.add_argument('--val_meta_data_path', default='../../D2_data/meta2', type=str, action='append', nargs='*')
parser.add_argument('--record_path', default='../../D2_data/record', type=str)
parser.add_argument('--multiprocess', default=12, type=int)
# set up network configuration.
# parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
# parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
# parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax', 'mse'], type=str)
# parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], type=str)
# parser.add_argument('--ohem_level', default=0, type=int,
#                     help='pick hard samples from (ohem_level * batch_size) proposals, must be > 1')
parser.add_argument('--data_format', default='wav', choices=['wav', 'npy'], type=str)
parser.add_argument('--audio_length', default=2.5, type=float)

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
    import generator

    # ==================================
    #       Get Train/Val.
    # ==================================
    trnlist, trnlb = toolkits.get_hike_datalist(meta_path=args.train_meta_data_path, data_path=args.train_data_path, mode=model_config['loss'])
    vallist, vallb = toolkits.get_hike_datalist(meta_path=args.val_meta_data_path, data_path=args.val_data_path, mode=model_config['loss'])

    input_length = int(args.audio_length * 25)
    num_class = len(score_rule)
    # construct the data generator.
    params = {'dim': (513, input_length, 1),
              'mp_pooler': toolkits.set_mp(processes=args.multiprocess),
              'nfft': 1024,
              'spec_len': input_length,
              'win_length': 1024,
              'hop_length': 640,
              'n_classes': num_class,
              'sampling_rate': 16000,
              'batch_size': args.batch_size,
              'shuffle': True,
              'normalize': True,
              'loss': model_config['loss'],
              'data_format': args.data_format
              }

    # Datasets
    partition = {'train': trnlist.flatten(), 'val': vallist.flatten()}
    labels = {'train': trnlb.flatten(), 'val': vallb.flatten()}

    # Generators
    wandb.init(project='vgg_speaker')
    trn_gen = generator.DataGenerator(partition['train'], labels['train'], **params)
    val_gen = generator.DataGenerator(partition['val'], labels['val'], **params)
    network = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                           num_class=params['n_classes'],
                                           mode='train', args=model_config)
    # # val data
    # val_data = [params['mp_pooler'].apply_async(ut.load_data,
    #                                 args=(ID, params['win_length'], params['sampling_rate'], params['hop_length'],
    #                                       params['nfft'], params['spec_len'], 'train', args.data_format)) for ID in partition['val']]
    # val_data = np.expand_dims(np.array([p.get() for p in val_data]), -1)

    # ==> load pre-trained model ???
    print(keras.backend.tensorflow_backend._get_available_gpus())

    if args.resume:
        print("Attempting to load", args.resume)
        if args.resume:
            if os.path.isfile(args.resume):
                network.load_weights(os.path.join(args.resume), by_name=True, skip_mismatch=True)
                print('==> successfully loading model {}.'.format(args.resume))
            else:
                raise ValueError("==> no checkpoint found at '{}'".format(args.resume))

    print(network.summary())
    print('==> gpu {} is, training {} images, classes: 0-{} '
          'loss: {}, aggregation: {}, ohemlevel: {}'.format(args.gpu, len(partition['train']), np.max(labels['train']),
                                                            model_config['loss'], model_config['aggregation_mode'],
                                                            model_config['ohem_level']))

    model_path, log_path = set_path(args, model_config)
    normal_lr = keras.callbacks.LearningRateScheduler(step_decay)
    # tbcallbacks = keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=False,
    #                                           update_freq=model_config['batch_size'] * 16)
    callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(model_path, 'weights-{epoch:02d}-{loss:.3f}.h5'),
                                                 monitor='loss',
                                                 mode='min',
                                                 save_best_only=True,
                                                 period=20,
                                                 ), normal_lr, WandbCallback()]

    if model_config['ohem_level'] > 1:     # online hard negative mining will be used
        candidate_steps = int(len(partition['train']) // model_config['batch_size'])
        iters_per_epoch = int(len(partition['train']) // (model_config['ohem_level']*model_config['batch_size']))

        ohem_generator = generator.OHEM_generator(network,
                                                  trn_gen,
                                                  candidate_steps,
                                                  model_config['ohem_level'],
                                                  model_config['batch_size'],
                                                  params['dim'],
                                                  params['n_classes']
                                                  )

        A = ohem_generator.next()   # for some reason, I need to warm up the generator

        network.fit_generator(generator.OHEM_generator(network, trn_gen, iters_per_epoch,
                                                       model_config['ohem_level'], model_config['batch_size'],
                                                       params['dim'], params['n_classes']),
                              steps_per_epoch=iters_per_epoch,
                              epochs=model_config['epochs'],
                              max_queue_size=10,
                              callbacks=callbacks,
                              use_multiprocessing=False,
                              workers=1,
                              verbose=1)

    else:
        if model_config['loss'] != 'mse':
            network.fit_generator(trn_gen,
                                  steps_per_epoch=int(len(partition['train'])//args.batch_size),
                                  epochs=model_config['epochs'],
                                  max_queue_size=10,
                                  validation_data=val_gen,
                                  validation_freq=1,
                                  callbacks=callbacks,
                                  use_multiprocessing=False,
                                  workers=1,
                                  verbose=1)
        else:
            network.fit_generator(trn_gen,
                                  steps_per_epoch=int(len(partition['train'])//model_config['batch_size']),
                                  epochs=model_config['epochs'],
                                  max_queue_size=10,
                                  validation_data=val_gen,
                                  validation_freq=1,
                                  callbacks=callbacks,
                                  use_multiprocessing=False,
                                  workers=1,
                                  verbose=1)



def step_decay(epoch):
    '''
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every step epochs.
    '''
    half_epoch = model_config['epochs'] // 2
    stage1, stage2, stage3 = int(half_epoch * 0.5), int(half_epoch * 0.8), half_epoch
    stage4 = stage3 + stage1
    stage5 = stage4 + (stage2 - stage1)
    stage6 = model_config['epochs']

    if model_config['warmup_ratio']:
        milestone = [2, stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [model_config['warmup_ratio'], 1.0, 0.1, 0.01, 1.0, 0.1, 0.01]
    else:
        milestone = [stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [1.0, 0.1, 0.01, 1.0, 0.1, 0.01]

    lr = 0.005
    init_lr = model_config['lr']
    stage = len(milestone)
    for s in range(stage):
        if epoch < milestone[s]:
            lr = init_lr * gamma[s]
            break
    print('Learning rate for epoch {} is {}.'.format(epoch + 1, lr))
    return np.float(lr)


def set_path(args, model_config):
    import datetime
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    loss = model_config['loss']
    net = model_config['net']
    batch_size = model_config['batch_size']
    bottleneck_dim = model_config['bottleneck_dim']

    if model_config['aggregation_mode'] == 'avg':
        exp_path = os.path.join(model_config['aggregation_mode']+'_{}'.format(loss),
                                '{}_{}_bs{}_bdim{}'.format(date, net, batch_size, bottleneck_dim))
    elif model_config['aggregation_mode'] == 'vlad':
        exp_path = os.path.join(model_config['aggregation_mode']+'_{}'.format(loss),
                                '{}_{}_bs{}_bdim{}'.format(date, net, batch_size, bottleneck_dim))
    elif model_config['aggregation_mode'] == 'gvlad':
        exp_path = os.path.join(model_config['aggregation_mode']+'_{}'.format(loss),
                                '{}_{}_bs{}_bdim{}'.format(date, net, batch_size, bottleneck_dim))
    else:
        raise IOError('==> unknown aggregation mode.')
    model_path = os.path.join(args.record_path, 'model', exp_path)
    log_path = os.path.join(args.record_path, 'log', exp_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    if not os.path.exists(log_path): os.makedirs(log_path)
    return model_path, log_path


if __name__ == "__main__":
    main()
