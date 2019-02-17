import argparse


class RequireInCmdline(object):
    pass


def _default(vals, key):
    v = vals.get(key)
    return None if v is RequireInCmdline else v


def _required(vals, key):
    return vals.get(key) is RequireInCmdline

def parse_args(init_vals, custom_parser=None):
    if custom_parser is None:
        f = argparse.ArgumentDefaultsHelpFormatter
        p = argparse.ArgumentParser(formatter_class=f)
    else:
        p = custom_parser

    p.add_argument('--data_dir',
                   default=_default(init_vals, 'data_dir'),
                   required=_required(init_vals, 'data_dir'),
                   help='Path to dataset')

    p.add_argument('--log_dir',
                   default=_default(init_vals, 'log_dir'),
                   required=_required(init_vals, 'log_dir'),
                   help="""Directory in which to write training
                   summarizes and checkpoints.""")
    p.add_argument('-m', '--mode', choices=['train', 'evaluate', 'predict'],
                   default=_default(init_vals, 'mode'),
                   required=_required(init_vals, 'mode'),
                   help="""Select the mode for estimator in terms 
                   of train or evaluate""")

    p.add_argument('-e', '--n_epochs', type=int,
                   default=_default(init_vals, 'n_epochs'),
                   required=_required(init_vals, 'n_epochs'),
                   help="Number of epochs to run")

    p.add_argument('-b', '--batch_size', type=int,
                   default=_default(init_vals, 'batch_size'),
                   required=_required(init_vals, 'batch_size'),
                   help="Size of each minibatch.")

    p.add_argument('-f', '--n_folds', type=int,
                   default=_default(init_vals, 'n_folds'),
                   required=_required(init_vals, 'n_folds'),
                   help='Number of n-fold cross-validations')

    p.add_argument('-t', '--AE_TYPE', choices=['item', 'user'],
                   default=_default(init_vals, 'AE_TYPE'),
                   required=_required(init_vals, 'AE_TYPE'),
                   help="""Select the type of AE in temrs of 
                   item or user""")

    p.add_argument('--precision', choices=['fp32', 'fp16'],
                   default=_default(init_vals, 'precision'),
                   required=_required(init_vals, 'precision'),
                   help="Select single of half precision arithmetic.")

    p.add_argument('--l2_lambda', type=float,
                   default=_default(init_vals, 'l2_lambda'),
                   required=_required(init_vals, 'l2_lambda'),
                   help='L2 regularization parameters.')

    p.add_argument('--display_every', type=int,
                   default=_default(init_vals, 'display_every'),
                   required=_required(init_vals, 'display_every'),
                   help='How often to print out information')

    p.add_argument('--test_mode', choices=['warm', 'cold_user', 'cold_item'],
                   default=_default(init_vals, 'test_mode'),
                   required=_required(init_vals, 'test_mode'),
                   help="""Select the test mode for estimator 
                   in terms of warm, cold_user and cold item""")


    FLAGS, unknown_args = p.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    vals = init_vals
    vals['data_dir'] = FLAGS.data_dir
    del FLAGS.data_dir
    vals['log_dir'] = FLAGS.log_dir
    del FLAGS.log_dir
    vals['n_epochs'] = FLAGS.n_epochs
    del FLAGS.n_epochs
    vals['batch_size'] = FLAGS.batch_size
    del FLAGS.batch_size
    vals['n_folds'] = FLAGS.n_folds
    del FLAGS.n_folds
    vals['AE_TYPE'] = FLAGS.AE_TYPE
    del FLAGS.AE_TYPE
    vals['precision'] = FLAGS.precision
    del FLAGS.precision
    vals['l2_lambda'] = FLAGS.l2_lambda
    del FLAGS.l2_lambda
    vals['display_every'] = FLAGS.display_every
    del FLAGS.display_every
    vals['dims'] = FLAGS.dims
    del FLAGS.dims
    vals['mode'] = FLAGS.mode
    del FLAGS.mode
    vals['test_mode'] = FLAGS.test_mode
    del FLAGS.test_mode


    return vals, FLAGS
