import numpy as np

import model
import argparse
import utils
import _pickle
import os
import shutil

utils.init()

default_args = {
    'mode': 'train',
    'batch_size': 1024,
    'lr': 1e-3,
    'precision': 'fp32',
    'n_epochs': 10000,
    'data_dir': './Input/ml-1m/ratings.dat',
    'log_dir': './log',
    'model_dir': './model',
    'display_every': 1000,
    'l2_lambda': 1e-3,
    'rank': 10,
    'eps': 0.1,
    'AE_TYPE': 'item',
    'height': None,
    'width': None,
    'n_folds': 1,
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
def _get_input(args, mode):
    meta_path = os.path.join(args['data_dir'], 'meta')
    meta = _pickle.loads(open(meta_path).read())

    args['height'] = meta['height']
    args['width']  = meta['width']

    print("[*] Input shape: %d x %d" % (args['height'], args['width']))

    dims = [int(i) for i in args['dims']]
    width = int(args['width'])
    dims = np.concatenate([[width], dims, [width]], 0)
    args['dims'] = dims

    return args


def train(args):
    args = _get_input(args, mode='train')
    RMSE, BEST_RMSE = [], 9999999

    for i in range(args['n_folds']):
        log_dir = args['log_dir']
        model_dir = args['model_dir']

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        os.makedirs(log_dir)

        # TRAINIG
        BEA = model.AE_CF(args)
        utils.train(BEA, args)

        # # VALIDATION
        # RMSE_VAL = utils.validate(BEA, args)
        # RMSE.append(RMSE_VAL)

        BEA.destroy_graph()

        # if RMSE_VAL < BEST_RMSE:
        #     BEST_RMSE = RMSE_VAL
        #     data = args['data']
        #
        #     # Remove the previous best model
        #     utils.remove_files(model_dir)
        #
        #     # Copy the best model to model directory
        #     utils.copy_files(log_dir, model_dir)
        #
        # utils.write_files(model_dir, data)


def evaluate(args):
    if args['model_dir'] is not None:
        args = _get_input(args, mode='val')
        BEA = model.AE_CF(args)
        _ = utils.validate(BEA, args)
        BEA.destroy_graph()

def predict(args):
    if args['model_dir'] is not None:
        args = _get_input(args, mode='val')
        BEA = model.AE_CF(args)
        _ = utils.predict(BEA, args)
        BEA.destroy_graph()

if args['mode'] == 'train':
    train(args)
elif args['mode'] == 'predict':
    predict(args)
else:
    evaluate(args)
