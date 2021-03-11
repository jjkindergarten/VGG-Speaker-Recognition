import os
import json
import numpy as np
import pandas as pd
from collections import Counter

def initialize_GPU(args):
    # Initialize GPUs
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def get_chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


def debug_generator(generator):
    import cv2
    import pdb
    G = generator.next()
    for i,img in enumerate(G[0]):
        path = '../sample/{}.jpg'.format(i)
        img = np.asarray(img[:,:,::-1] + 128.0, dtype='uint8')
        cv2.imwrite(path, img)


# set up multiprocessing
def set_mp(processes=8):
    import multiprocessing as mp

    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except:
        pass

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool


# vggface2 dataset
def get_vggface2_imglist(args):
    def get_datalist(s):
        file = open('{}'.format(s), 'r')
        datalist = file.readlines()
        imglist = []
        labellist = []
        for i in datalist:
            linesplit = i.split(' ')
            imglist.append(linesplit[0])
            labellist.append(int(linesplit[1][:-1]))
        return imglist, labellist

    print('==> calculating image lists...')
    # Prepare training data.
    imgs_list_trn, lbs_list_trn = get_datalist(args.trn_meta)
    imgs_list_trn = [os.path.join(args.data_path, i) for i in imgs_list_trn]
    imgs_list_trn = np.array(imgs_list_trn)
    lbs_list_trn = np.array(lbs_list_trn)

    # Prepare validation data.
    imgs_list_val, lbs_list_val = get_datalist(args.val_meta)
    imgs_list_val = [os.path.join(args.data_path, i) for i in imgs_list_val]
    imgs_list_val = np.array(imgs_list_val)
    lbs_list_val = np.array(lbs_list_val)

    return imgs_list_trn, lbs_list_trn, imgs_list_val, lbs_list_val


def get_imagenet_imglist(args, trn_meta_path='', val_meta_path=''):
    with open(trn_meta_path) as f:
        strings = f.readlines()
        trn_list = np.array([os.path.join(args.data_path, '/'.join(string.split()[0].split(os.sep)[-4:]))
                             for string in strings])
        trn_lb = np.array([int(string.split()[1]) for string in strings])
        f.close()

    with open(val_meta_path) as f:
        strings = f.readlines()
        val_list = np.array([os.path.join(args.data_path, '/'.join(string.split()[0].split(os.sep)[-4:]))
                             for string in strings])
        val_lb = np.array([int(string.split()[1]) for string in strings])
        f.close()
    return trn_list, trn_lb, val_list, val_lb


def get_voxceleb2_datalist(args, path):
    with open(path) as f:
        strings = f.readlines()
        audiolist = np.array([os.path.join(args.data_path, string.split()[0]) for string in strings])
        labellist = np.array([int(string.split()[1]) for string in strings])
        f.close()
    return audiolist, labellist

def get_hike_datalist(args, path):
    def assgin_category(score):
        if score <= 4:
            return 0
        elif score >= 7:
            return 1
        else:
            return 2
    category = args.category.split('_')
    with open(path) as f:
        meta_list = json.load(f)
        audiolist = np.array([os.path.join(args.data_path, i[0]) for i in meta_list if i[2] in category])
        labellist = np.array([assgin_category(i[1]) for i in meta_list if i[2] in category])
        f.close()
    print('class weight: {}'.format(Counter(labellist)))
    return audiolist, labellist

def get_hike_datalist2(args, path):
    category = args.category.split('_')
    with open(path) as f:
        meta_list = json.load(f)
        audiolist = np.array([os.path.join(args.data_path, i[0]) for i in meta_list if i[2] in category])
        scorelist = np.array([i[1] for i in meta_list if i[2] in category])
        scorelist = (scorelist-5)/10
        f.close()
    return audiolist, scorelist

def calculate_eer(y, y_score):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def sync_model(src_model, tgt_model):
    print('==> synchronizing the model weights.')
    params = {}
    for l in src_model.layers:
        params['{}'.format(l.name)] = l.get_weights()

    for l in tgt_model.layers:
        if len(l.get_weights()) > 0:
            l.set_weights(params['{}'.format(l.name)])
    return tgt_model

def get_content_score(path, score, args):
    category = args.category.split('_')
    with open(os.path.join(path, 'hike_val_{}.json'.format(args.seed)), 'r') as f:
        meta_list_val = json.load(f)
        contentlist_val = [i[3] for i in meta_list_val if i[2] in category]
    with open(os.path.join(path, 'hike_test_{}.json'.format(args.seed)), 'r') as f:
        meta_list_test = json.load(f)
        contentlist_test = [i[3] for i in meta_list_test if i[2] in category]

    contentlist = np.array(contentlist_val + contentlist_test)

    print(len(score))
    print(len(contentlist))
    assert len(score) == len(contentlist)
    df = np.hstack([contentlist.reshape(-1,1), score])
    df = pd.DataFrame(data=df, columns=['content', 'score_predict', 'score_true'])
    df = df.sort_values(by='content', ignore_index=True)
    df = df.astype({'score_predict': 'float', 'score_true': 'float'})
    df_content = df.groupby(['content']).mean()
    print(df_content)
    print('mse by content: ', np.square(np.subtract(df_content.values[:,0], df_content.values[:,1])).mean())
