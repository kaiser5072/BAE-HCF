import tensorflow as tf
import numpy as np

import h5py
import tqdm
import time
import utils

class _PrefillStagingAreasHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        enqueue_ops = tf.get_collection('STAGING_AREA_PUTS')
        for i in range(len(enqueue_ops)):
            session.run(enqueue_ops[:i+1])


class _LogSessionRunHook(tf.train.SessionRunHook):
    def __init__(self, global_batch_size, num_records, display_every=1):
        self.global_batch_size = global_batch_size
        self.num_records = num_records
        self.display_every = display_every
    def after_create_session(self, session, coord):
        print('\n|  Step  Epoch Step/sec  Loss TotalLoss  Time  LR|')
        self.elapsed_secs = 0.
        self.count = 0
        self.t0 = time.time()
    def before_run(self, run_context):
        self.t1 = time.time()
        return tf.train.SessionRunArgs(
            fetches=['step_update:0', 'loss:0', 'total_loss:0', 'lr:0'])
    def after_run(self, run_context, run_values):
        self.elapsed_secs += time.time() - self.t1
        self.process_secs = time.time() - self.t0
        self.count += 1
        global_step, loss, total_loss, lr = run_values.results
        print_step = global_step + 1
        if print_step == 1 or print_step % self.display_every == 0:
            dt = self.elapsed_secs / self.count
            img_per_sec = 1 / dt
            epoch = print_step * self.global_batch_size / self.num_records
            print("|%6i %7.1f %5.1f %6.3f %7.3f %7.1f   %6.6f|" %
                  (print_step, epoch, img_per_sec, np.sqrt(loss), np.sqrt(total_loss), self.process_secs, lr))
            self.elapsed_secs = 0.
            self.count = 0

def train(infer_func, params):
    batch_size       = params['batch_size']
    n_epochs         = params['n_epochs']
    data_dir         = params['data_dir']
    log_dir          = params['log_dir']
    height           = params['height']
    width            = params['width']
    AE_type          = params['AE_TYPE']
    n_features       = params['n_features']
    prefetch_size    = params['prefetch_size']
    display_step     = params['display_every']

    decay_steps = n_epochs // 10
    nstep = display_step# * height // batch_size
    nupdate = n_epochs * height // batch_size

    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 32
    est = tf.estimator.Estimator(
        model_fn=infer_func._BAE_model_fn,
        model_dir=log_dir,
        params={
            'height': params['batch_size'],
            'width':  params['width'],
            'drop_rate': 0.5
        },
        config=tf.estimator.RunConfig(
            session_config=config,
            save_checkpoints_secs=None,
            save_checkpoints_steps=nstep,
            keep_checkpoint_every_n_hours=3))

    training_hooks = [_PrefillStagingAreasHook(),
                      _LogSessionRunHook(batch_size, height, display_step)]

    input_func = lambda: utils.data_set(data_dir, batch_size, prefetch_size, width, n_features,
                                        mode='train', AE_type=AE_type, is_repeat=True)


    print("\n\n TRAINING\n")
    try:
        est.train(
            input_fn=input_func,
            max_steps=nupdate,
            hooks=training_hooks)
    except KeyboardInterrupt:
        print("Keyboard interrupt")

#TODO: Real-time evaluation
def validate(infer_func, params):
    data_dir        = params['data_dir']
    log_dir         = params['log_dir'] #if params['mode'] == 'train' else params['model_dir']
    width           = params['width']
    n_features      = params['n_features']
    batch_size      = params['batch_size']
    prefetch_size   = params['prefetch_size']

    config = tf.ConfigProto()
    config.gpu_options.force_gpu_compatible = True
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 32

    est = tf.estimator.Estimator(
        model_fn=infer_func._BAE_model_fn,
        model_dir=log_dir,
        params={
            'height': params['batch_size'],
            'width' : params['width'],
            'drop_rate': 1
        },
        config=tf.estimator.RunConfig(
            session_config=config,
            save_checkpoints_secs=None,
            save_checkpoints_steps=None,
            keep_checkpoint_every_n_hours=3))

    input_func = lambda: utils.data_set(data_dir, batch_size, prefetch_size, width, n_features,
                                        mode='eval')

    print("\n\nEVALUATE\n")
    try:
        eval_result = est.evaluate(
            input_fn = input_func)

        print('\n [*] RMSE_TRAIN: %.4f' % eval_result['rmse_tr'])
        print(' [*] RMSE_TEST: %.4f' % eval_result['rmse_te'])

        return eval_result['rmse_te']

    except KeyboardInterrupt:
        print("Keyboard interrupt")


def predict(infer_func, params):
    data_dir        = params['data_dir']
    log_dir         = params['log_dir'] #if params['mode'] == 'train' else params['model_dir']
    height          = params['height']
    width           = params['width']
    AE_type         = params['AE_TYPE']
    n_features      = params['n_features']
    batch_size      = params['batch_size']
    prefetch_size   = params['prefetch_size']

    config = tf.ConfigProto()
    config.gpu_options.force_gpu_compatible = True
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 32

    if params['test_mode'] == 'warm':
        test_idx = h5py.File('warm_index.h5py', 'r')
    elif params['test_mode'] == 'cold_user':
        test_idx = h5py.File('cold_user_index.h5py', 'r')
    else:
        test_idx = h5py.File('cold_item_index.h5py', 'r')
    item_idx = test_idx['idx']

    est = tf.estimator.Estimator(
        model_fn=infer_func._BAE_model_fn,
        model_dir=log_dir,
        params={
            'height': params['batch_size'],
            'width' : params['width'],
            'drop_rate': 1.0
        },
        config=tf.estimator.RunConfig(
            session_config=config,
            save_checkpoints_secs=None,
            save_checkpoints_steps=None,
            keep_checkpoint_every_n_hours=3))

    input_func = lambda: utils.data_set(data_dir, batch_size, prefetch_size, width, n_features,
                                        mode='predict', AE_type=AE_type, is_repeat=False)

    print("\n\n PREDICT\n")
    try:
        eval_result = est.predict(
            input_fn=input_func)

        preds = np.zeros((height, len(item_idx)))
        with tqdm.tqdm(total=height) as pbar:
            for i, pred in enumerate(eval_result):
                _pred = pred['preds'][item_idx]

                preds[i, :] = _pred
                pbar.update(1)

        print('\n')
        return preds

    except KeyboardInterrupt:
        print("Keyboard interrupt")