import tensorflow as tf
import numpy as np

import tqdm
import time
import utils
from scipy.stats import rankdata

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
        print('\n|  Step  Epoch Step/sec  Loss TotalLoss  Time  |')
        self.elapsed_secs = 0.
        self.count = 0
        self.t0 = time.time()
    def before_run(self, run_context):
        self.t1 = time.time()
        return tf.train.SessionRunArgs(
            fetches=['step_update:0', 'loss:0', 'total_loss:0'])
    def after_run(self, run_context, run_values):
        self.elapsed_secs += time.time() - self.t1
        self.process_secs = time.time() - self.t0
        self.count += 1
        global_step, loss, total_loss = run_values.results
        print_step = global_step + 1
        if print_step == 1 or print_step % self.display_every == 0:
            dt = self.elapsed_secs / self.count
            img_per_sec = 1 / dt
            epoch = print_step * self.global_batch_size / self.num_records
            print("|%6i %7.1f %5.1f %6.3f %7.3f %7.1f   |" %
                  (print_step, epoch, img_per_sec, np.sqrt(loss), np.sqrt(total_loss), self.process_secs))
            self.elapsed_secs = 0.
            self.count = 0

def train(infer_func, params):
    batch_size       = params['batch_size']
    n_epochs         = params['n_epochs']
    data_dir         = params['data_dir']
    log_dir          = params['log_dir']
    height           = params['height']
    width            = params['width']
    prefetch_size    = params['prefetch_size']
    display_step     = params['display_every']
    checkpoints_secs = params['checkpoints_secs']

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
            'width':  params['width']
        },
        config=tf.estimator.RunConfig(
            session_config=config,
            save_checkpoints_secs=checkpoints_secs,
            save_checkpoints_steps=nstep,
            keep_checkpoint_every_n_hours=3))

    training_hooks = [_PrefillStagingAreasHook(),
                      _LogSessionRunHook(batch_size, height, display_step)]

    input_func = lambda: utils.data_set(data_dir, batch_size, prefetch_size, width,
                                        mode='train')


    print("\n\n TRAINING\n")
    try:
        est.train(
            input_fn=input_func,
            max_steps=nupdate,
            hooks=training_hooks)
    except KeyboardInterrupt:
        print("Keyboard interrupt")

def validate(infer_func, params):
    data_dir        = params['data_dir']
    log_dir         = params['log_dir'] #if params['mode'] == 'train' else params['model_dir']
    width           = params['width']
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
            'width' : params['width']
        },
        config=tf.estimator.RunConfig(
            session_config=config,
            save_checkpoints_secs=None,
            save_checkpoints_steps=None,
            keep_checkpoint_every_n_hours=3))

    input_func = lambda: utils.data_set(data_dir, batch_size, prefetch_size, width,
                                        mode='eval')

    print("\n\nEVALUATE\n")
    try:
        eval_result = est.evaluate(
            input_fn = input_func)

        print('\n [*] RECALL: %.4f' % eval_result['recall'])
        print(' [*] RMSE: %.4f' % eval_result['rmse'])
        return eval_result['recall']
    except KeyboardInterrupt:
        print("Keyboard interrupt")


def predict(infer_func, params):
    data_dir        = params['data_dir']
    log_dir         = params['log_dir'] #if params['mode'] == 'train' else params['model_dir']
    height          = params['height']
    width           = params['width']
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
            'width' : params['width']
        },
        config=tf.estimator.RunConfig(
            session_config=config,
            save_checkpoints_secs=None,
            save_checkpoints_steps=None,
            keep_checkpoint_every_n_hours=3))

    input_func = lambda: utils.data_set(data_dir, batch_size, prefetch_size, width,
                                        mode='eval')

    print("\n\n PREDICT\n")
    try:
        eval_result = est.predict(
            input_fn=input_func)

        preds, ratingTest, mask = [], [], []
        with tqdm.tqdm(total=height) as pbar:
            for pred in eval_result:
                _pred = pred['preds']
                _rating = pred['ratingTest']
                _mask = pred['mask']

                preds.append(_pred)
                ratingTest.append(_rating)
                mask.append(_mask)
                pbar.update(1)


        recall = utils.get_recall(ratingTest, preds, mask, np.arange(50, 550, 50))
        print("\n [*] RECALL: %.4f" % recall)

    except KeyboardInterrupt:
        print("Keyboard interrupt")