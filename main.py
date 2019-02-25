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

default_args = {
    'mode': 'train',
    'test_mode': 'warm',
    'batch_size': 1024,
    'lr': 1e-3,
    'precision': 'fp32',
    'n_epochs': 10,
    'log_dir': './log',
    'model_dir': './model',
    'display_every': 1000,
    'l2_lambda': 1e-4,
    'AE_TYPE': 'item',
    'height': None,
    'width': None,
    'prefetch_size': 20,
    'checkpoints_secs': None,
    'device': '/gpu:0'
}

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)

parser.add_argument('--dims', '--list', nargs='+',
                    default=[500, 500, 1500],
                    required=True,
                    help='The number of units')

args, flags = utils.parse_args(default_args, parser)

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
    print(args['l2_lambda'])

    for i in range(args['n_folds']):
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
    if args['model_dir'] is not None:
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
