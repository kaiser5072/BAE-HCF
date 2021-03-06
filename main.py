import numpy as np

import model
import argparse
import utils
import cPickle
import os
import shutil

from utils import Option
opt = Option('./config.json')

utils.init()

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)

args, flags = utils.parse_args(opt, parser)

# TODO: Combine lines to load rating data and save dictionary.
def _get_input(args):
    meta_path = os.path.join(args['data_dir'], 'meta')
    meta = cPickle.loads(open(meta_path).read())

    meta_path = os.path.join('./Input', 'meta')
    meta2 = cPickle.loads(open(meta_path).read())

    if args['AE_TYPE'] == 'item':
        args['height']     = meta['n_item_height']
        args['width']      = meta2['n_user']
        args['n_features'] = int(meta2['n_content_item'])
    else:
        args['height']     = meta['n_user_height']
        args['width']      = meta2['n_item']
        args['n_features'] = int(meta2['n_content_user'])

    print("[*] Input shape: %d x %d" % (args['height'], args['width']))

    dims = [int(i) for i in args['dims']]
    width = int(args['width'])
    dims = np.concatenate([[width], dims, [width]], 0)
    args['dims'] = dims

    return args


def train(args):
    args = _get_input(args)

    log_dir = args['log_dir']

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    os.makedirs(log_dir)

    # TRAINIG
    BEA = model.AE_CF(args)
    utils.train(BEA, args)

    BEA.destroy_graph()

def evaluate(args):
    if args['model_dir'] is not None:
        args = _get_input(args)
        BEA = model.AE_CF(args)
        _ = utils.validate(BEA, args)
        BEA.destroy_graph()

def predict(args):
    meta_path = os.path.join(args['data_dir'], 'meta')
    meta = cPickle.loads(open(meta_path).read())

    args = _get_input(args)
    BEA = model.AE_CF(args)
    preds = utils.predict(BEA, args)
    if args['AE_TYPE'] == 'item':
        preds = np.transpose(preds)

    utils.get_eval(preds, args['test_mode'], meta)
    BEA.destroy_graph()

if args['mode'] == 'train':
    train(args)
elif args['mode'] == 'predict':
    predict(args)
else:
    evaluate(args)
