from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import keras
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
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--data_path', default='../../../D2_data/audio_data_OTR', type=str)
parser.add_argument('--meta_data_path', default='../../../D2_data/meta2', type=str)
parser.add_argument('--record_path', default='../../../D2_data/record', type=str)
parser.add_argument('--multiprocess', default=12, type=int)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=10, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--warmup_ratio', default=0, type=float)
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax', 'regression'], type=str)
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], type=str)
parser.add_argument('--ohem_level', default=0, type=int,
                    help='pick hard samples from (ohem_level * batch_size) proposals, must be > 1')
parser.add_argument('--seed', default=2, type=int, help='seed for which dataset to use')
parser.add_argument('--data_format', default='wav', choices=['wav', 'npy'], type=str)
parser.add_argument('--audio_length', default=2.5, type=float)
global args
args = parser.parse_args()

def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model
    import generator

    # ==================================
    #       Get Train/Val.
    # ==================================

    if args.loss != 'regression':
        trnlist, trnlb = toolkits.get_hike_datalist(args, path=os.path.join(args.meta_data_path,
                                                                            'hike_train_{}.json'.format(args.seed)))
        vallist, vallb = toolkits.get_hike_datalist(args, path=os.path.join(args.meta_data_path,
                                                                            'hike_val_{}.json'.format(args.seed)))
    else:
        trnlist, trnlb = toolkits.get_hike_datalist2(args, path=os.path.join(args.meta_data_path,
                                                                            'hike_train_{}.json'.format(args.seed)))
        vallist, vallb = toolkits.get_hike_datalist2(args, path=os.path.join(args.meta_data_path,
                                                                            'hike_val_{}.json'.format(args.seed)))

    input_length = int(args.audio_length * 100)
    # construct the data generator.
    params = {'dim': (257, input_length, 1),
              'mp_pooler': toolkits.set_mp(processes=args.multiprocess),
              'nfft': 512,
              'spec_len': input_length,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 2,
              'sampling_rate': 16000,
              'batch_size': args.batch_size,
              'shuffle': True,
              'normalize': True,
              'loss': args.loss,
              'data_format': args.data_format
              }

    # Datasets
    partition = {'train': trnlist.flatten(), 'val': vallist.flatten()}
    labels = {'train': trnlb.flatten(), 'val': vallb.flatten()}

    # Generators
    trn_gen = generator.DataGenerator(partition['train'], labels['train'], **params)
    network = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                           num_class=params['n_classes'],
                                           mode='train', args=args)
    # val data
    val_data = [params['mp_pooler'].apply_async(ut.load_data,
                                    args=(ID, params['win_length'], params['sampling_rate'], params['hop_length'],
                                          params['nfft'], params['spec_len'], 'train', args.data_format)) for ID in partition['val']]
    val_data = np.expand_dims(np.array([p.get() for p in val_data]), -1)

    # ==> load pre-trained model ???
    print(keras.backend.tensorflow_backend._get_available_gpus())

    if args.resume:
        print("Attempting to load", args.resume)
        if args.resume:
            if os.path.isfile(args.resume):
                # by_name=True, skip_mismatch=True
                # https://github.com/WeidiXie/VGG-Speaker-Recognition/issues/46
                network.load_weights(os.path.join(args.resume), by_name=True, skip_mismatch=True)
                print('==> successfully loading model {}.'.format(args.resume))
            else:
                raise ValueError("==> no checkpoint found at '{}'".format(args.resume))

    print(network.summary())
    print('==> gpu {} is, training {} images, classes: 0-{} '
          'loss: {}, aggregation: {}, ohemlevel: {}'.format(args.gpu, len(partition['train']), np.max(labels['train']),
                                                            args.loss, args.aggregation_mode, args.ohem_level))

    model_path, log_path = set_path(args)
    normal_lr = keras.callbacks.LearningRateScheduler(step_decay)
    tbcallbacks = keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=False,
                                              update_freq=args.batch_size * 16)
    callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(model_path, 'weights-{epoch:02d}-{loss:.3f}.h5'),
                                                 monitor='loss',
                                                 mode='min',
                                                 save_best_only=True,
                                                 period=5,
                                                 ),
                 normal_lr, tbcallbacks]

    if args.ohem_level > 1:     # online hard negative mining will be used
        candidate_steps = int(len(partition['train']) // args.batch_size)
        iters_per_epoch = int(len(partition['train']) // (args.ohem_level*args.batch_size))

        ohem_generator = generator.OHEM_generator(network,
                                                  trn_gen,
                                                  candidate_steps,
                                                  args.ohem_level,
                                                  args.batch_size,
                                                  params['dim'],
                                                  params['n_classes']
                                                  )

        A = ohem_generator.next()   # for some reason, I need to warm up the generator

        network.fit_generator(generator.OHEM_generator(network, trn_gen, iters_per_epoch,
                                                       args.ohem_level, args.batch_size,
                                                       params['dim'], params['n_classes']),
                              steps_per_epoch=iters_per_epoch,
                              epochs=args.epochs,
                              max_queue_size=10,
                              callbacks=callbacks,
                              use_multiprocessing=False,
                              workers=1,
                              verbose=1)

    else:
        if args.loss != 'regression':
            network.fit_generator(trn_gen,
                                  steps_per_epoch=int(len(partition['train'])//args.batch_size),
                                  epochs=args.epochs,
                                  max_queue_size=10,
                                  validation_data=(val_data, keras.utils.to_categorical(labels['val'], num_classes=params['n_classes'])),
                                  validation_freq=1,
                                  callbacks=callbacks,
                                  use_multiprocessing=False,
                                  workers=1,
                                  verbose=1)
        else:
            network.fit_generator(trn_gen,
                                  steps_per_epoch=int(len(partition['train'])//args.batch_size),
                                  epochs=args.epochs,
                                  max_queue_size=10,
                                  validation_data=(val_data, labels['val']),
                                  validation_freq=1,
                                  callbacks=callbacks,
                                  use_multiprocessing=False,
                                  workers=1,
                                  verbose=1)

    # testlist, testlb = toolkits.get_hike_datalist(args, path='../meta/hike_test_{}.json'.format(args.seed))
    #
    # test_data = [params['mp_pooler'].apply_async(ut.load_data,
    #                                 args=(ID, params['win_length'], params['sampling_rate'], params['hop_length'],
    #                                       params['nfft'], params['spec_len'])) for ID in testlist]
    # test_data = np.expand_dims(np.array([p.get() for p in test_data]), -1)
    #
    # v = network.predict(test_data)
    # v = ((v<0.5)*1)[:,0]
    # acc = sum(v==vallb)/len(vallb)
    # print('test data predict accuracy is {}'.format(acc))


def step_decay(epoch):
    '''
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every step epochs.
    '''
    half_epoch = args.epochs // 2
    stage1, stage2, stage3 = int(half_epoch * 0.5), int(half_epoch * 0.8), half_epoch
    stage4 = stage3 + stage1
    stage5 = stage4 + (stage2 - stage1)
    stage6 = args.epochs

    if args.warmup_ratio:
        milestone = [2, stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [args.warmup_ratio, 1.0, 0.1, 0.01, 1.0, 0.1, 0.01]
    else:
        milestone = [stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [1.0, 0.1, 0.01, 1.0, 0.1, 0.01]

    lr = 0.005
    init_lr = args.lr
    stage = len(milestone)
    for s in range(stage):
        if epoch < milestone[s]:
            lr = init_lr * gamma[s]
            break
    print('Learning rate for epoch {} is {}.'.format(epoch + 1, lr))
    return np.float(lr)


def set_path(args):
    import datetime
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    if args.aggregation_mode == 'avg':
        exp_path = os.path.join(args.aggregation_mode+'_{}'.format(args.loss),
                                '{0}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                'lr{args.lr}_bdim{args.bottleneck_dim}_'
                                'ohemlevel{args.ohem_level}_seed{args.seed}'.format(date, args=args))
    elif args.aggregation_mode == 'vlad':
        exp_path = os.path.join(args.aggregation_mode+'_{}'.format(args.loss),
                                '{0}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                'lr{args.lr}_vlad{args.vlad_cluster}_'
                                'bdim{args.bottleneck_dim}_'
                                'ohemlevel{args.ohem_level}_seed{args.seed}'.format(date, args=args))
    elif args.aggregation_mode == 'gvlad':
        exp_path = os.path.join(args.aggregation_mode+'_{}'.format(args.loss),
                                '{0}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                'lr{args.lr}_vlad{args.vlad_cluster}_ghost{args.ghost_cluster}_'
                                'bdim{args.bottleneck_dim}_'
                                'ohemlevel{args.ohem_level}_seed{args.seed}'.format(date, args=args))
    else:
        raise IOError('==> unknown aggregation mode.')
    model_path = os.path.join(args.record_path, 'model', exp_path)
    log_path = os.path.join(args.record_path, 'log', exp_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    if not os.path.exists(log_path): os.makedirs(log_path)
    return model_path, log_path


if __name__ == "__main__":
    main()
