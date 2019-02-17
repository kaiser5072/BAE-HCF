import tensorflow as tf
import numpy as np

import h5py
import tqdm
import time
import utils
import os

from scipy.sparse import csr_matrix
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
    AE_type          = params['AE_type']
    n_features       = params['n_features']
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
            'width':  params['width'],
            'drop_rate': 0.5
        },
        config=tf.estimator.RunConfig(
            session_config=config,
            save_checkpoints_secs=checkpoints_secs,
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
    AE_type         = params['AE_type']
    n_features      = params['n_features']
    batch_size      = params['batch_size']
    prefetch_size   = params['prefetch_size']

    config = tf.ConfigProto()
    config.gpu_options.force_gpu_compatible = True
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 32

    warm_idx = h5py.File('warm_index.h5py', 'r')
    item_idx = warm_idx['idx']

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
                                        mode='val', AE_type=AE_type, is_repeat=False)

    print("\n\n PREDICT\n")
    try:
        eval_result = est.predict(
            input_fn=input_func)

        max_user = np.min((height, 100000))
        preds, target = np.zeros((max_user, len(item_idx))), np.zeros((max_user, len(item_idx)), dtype=np.int8)
        mask = np.zeros((max_user, len(item_idx)), dtype=np.int8)
        with tqdm.tqdm(total=height) as pbar:
            for i, pred in enumerate(eval_result):
                _pred = pred['preds'][item_idx]
                _target = pred['ratingTest'][item_idx]
                _mask = pred['ratingTrain'][item_idx]

                preds[i, :] = _pred
                target[i, :] = _target
                mask[i, :] = _mask
                pbar.update(1)

                if i >= max_user-1:
                    break

        print('\n')
        targets = csr_matrix(target)
        masks   = csr_matrix(mask)
        del target, mask, eval_result
        recalls = get_recall(targets, preds, masks, np.arange(50, 550, 50))

        for k, recall in zip(np.arange(50, 550, 50), recalls):
            print("[*] RECALL@%d: %.4f" % (k, recall))

    except KeyboardInterrupt:
        print("Keyboard interrupt")


def get_recall(target, preds, mask, n_recalls):
    # ratingTest[:, [1, 0]] = ratingTest[:, [0, 1]]


    # temp = np.zeros((16980, 5551))
    # temp[(ratingTest[:, 0], ratingTest[:, 1])] = 1
    # preds       = np.transpose(preds)
    # target      = np.transpose(ratingTest)
    # mask        = np.transpose(mask)
    preds  = np.asarray(preds)
    mask   = mask.toarray()
    print(np.sort(preds[0, :])[::-1][:100])
    print(np.sort(preds[0, :] * target[0].toarray()[0])[::-1])

    preds       = preds * (1-mask) - 100 * mask
    non_zero_idx = np.asarray(target.sum(axis=1)).flatten() != 0
    #
    del mask
    preds   = preds[non_zero_idx, :]
    target  = target[non_zero_idx]

    # pred_user_interest = pred_user_interest * test_mask + (1 - test_mask) * (-100)
    preds = get_order_array(preds)

    recall = []
    for i in n_recalls:
        pred_user_interest = preds <= i

        match_interest  = target.multiply(pred_user_interest)
        num_match       = np.sum(match_interest, axis=1, dtype=np.float32)
        num_interest    = target.sum(axis=1)

        user_recall = num_match / num_interest
        recall.append(np.average(user_recall))

    return recall

def get_order_array(list):
    order = np.empty(list.shape, dtype=int)
    for k, row in enumerate(list):
        order[k] = rankdata(-row, method='ordinal') - 1

    return order