import os
import json
import numpy as np
import pandas as pd
from collections import Counter

score_rule_file = '../../model_config/classification_rule.json'
filter_rule_file = '../../model_config/filter_rule.json'
with open(score_rule_file, 'r') as f:
    score_rule = json.load(f)
    f.close()
with open(filter_rule_file, 'r') as f:
    filter_rule = json.load(f)
    f.close()

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

def assign_category(score_rule, score):
    assert 'low' in score_rule
    assert 'high' in score_rule
    if (score <= score_rule['low']['max']) and (score > score_rule['low']['min']):
        return 0
    elif (score <= score_rule['high']['max']) and (score > score_rule['low']['min']):
        return 1
    elif 'medium' in score_rule:
        if (score <= score_rule['medium']['max']) and (score > score_rule['medium']['min']):
            return 2

    raise ValueError('{} cannot be categoried!'.format(score))

def score_filter(filter_rule, score):
    for rule in filter_rule:
        if (score >= min(rule)) and (score <= max(rule)):
            return True
    else:
        return False

def get_hike_datalist(meta_paths, data_paths, mode='mse'):
    assert len(meta_paths) == len(data_paths)
    audiolist = []
    labellist = []
    for i in range(len(meta_paths)):
        meta_path = meta_paths[i]
        data_path = data_paths[i]
        with open(meta_path) as f:
            meta_list = json.load(f)
            audiolist += [os.path.join(data_path, i[0]) for i in meta_list if score_filter(filter_rule, i[2])]
            if mode != 'mse':
                labellist += [assign_category(score_rule, i[2]) for i in meta_list if score_filter(filter_rule, i[2])]
            else:
                labellist += [i[2] for i in meta_list if i[2] in score_filter(filter_rule, i[2])]
            f.close()
    audiolist = np.array(audiolist)
    labellist = np.array(labellist)
    if mode == 'mse':
        labellist = (labellist - 5) / 10
    else:
        print('class weight: {}'.format(Counter(labellist)))
    return audiolist, labellist

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

def get_content_score(path, score):
    with open(path, 'r') as f:
        meta_list = json.load(f)
        contentlist = np.array([i[3] for i in meta_list if score_filter(filter_rule, i[2])])

    assert len(score) == len(contentlist)
    df = np.hstack([contentlist.reshape(-1,1), score])
    df = pd.DataFrame(data=df, columns=['content', 'score_predict', 'score_true'])
    df = df.sort_values(by='content', ignore_index=True)
    df = df.astype({'score_predict': 'float', 'score_true': 'float'})
    return df

def get_content_prob(path, classification):
    with open(path, 'r') as f:
        meta_list = json.load(f)
        contentlist = np.array([i[3] for i in meta_list if score_filter(filter_rule, i[2])])

    assert len(classification) == len(contentlist)
    df = np.hstack([contentlist.reshape(-1,1), classification])
    df = pd.DataFrame(data=df, columns=['content', 'prob_0', 'prob_1', 'true_label'])
    return df

