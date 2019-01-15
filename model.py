import tensorflow as tf
import numpy as np
from tensorflow.python.ops import data_flow_ops

def _stage(tensors):
    """Stages the given tensors in a StagingArea for asynchronous put/get
    """
    stage_area = data_flow_ops.StagingArea(
        dtypes = [tensor.dtype for tensor in tensors],
        shapes = [tensor.get_shape() for tensor in tensors])

    put_op = stage_area.put(tensors)
    get_tensors = stage_area.get()
    tf.add_to_collection('STAGING_AREA_PUTS', put_op)
    return put_op, get_tensors

class AE_CF(object):
    def __init__(self, params):
        self.dims          = params['dims']
        self.n_epochs      = params['n_epochs']
        self.n_features    = params['n_features']
        self.batch_size    = params['batch_size']
        self.lr_init       = params['lr']
        self.l2_lambda     = params['l2_lambda']
        self.rank          = params['rank']
        self.eps           = params['eps']
        self.device        = params['device']
        self.log_dir       = params['log_dir']
        self.prefetch_size = params['prefetch_size']

        self.dtype = tf.float16 if params['precision'] == 'fp16' else tf.float32
        self.n_layer = len(self.dims) - 1

    def builder(self, inputs, sides):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)
        h = inputs
        
        prev_dim = self.dims[0]
        for i in range(1, self.n_layer):
            with tf.variable_scope('layer%d'%i):
                w = tf.get_variable('weight', shape=[prev_dim, self.dims[i]],
                                              trainable=True,
                                              initializer=w_init,
                                              dtype=self.dtype)
                b = tf.get_variable('biases', shape=[self.dims[i]],
                                              trainable=True,
                                              initializer=b_init,
                                              dtype=self.dtype)
                s = tf.get_variable('weight_s', shape=[self.n_features, self.dims[i]],
                                                trainable=True,
                                                initializer=w_init,
                                                dtype=self.dtype)

            if i == 1 and self.n_layer != 2:
                # if np.random.uniform(0, 1, 1) < 0.5:
                h = tf.sparse.matmul(inputs, w) + b + tf.sparse.matmul(sides, s)
                # else:
                #     h = b + tf.sparse.matmul(sides, s)
                # h = tf.layers.batch_normalization(h)
                h = tf.nn.relu(h)

            elif self.n_layer == 2:
                # if np.random.uniform(0, 1, 1) < 0.5:
                h = tf.sparse.matmul(inputs, w) + b + tf.sparse.matmul(sides, s)
                # else:
                #     h = b + tf.sparse.matmul(sides, s)
                # h = tf.layers.batch_normalization(h)
                h = tf.nn.tanh(h)

            elif i == (self.n_layer-1):
                h = tf.matmul(h ,w) + b + tf.sparse.matmul(sides, s)
                # h = tf.layers.batch_normalization(h)
                h = tf.nn.tanh(h)

            else:
                h = tf.matmul(h, w) + b + tf.sparse.matmul(sides, s)
                # h = tf.layers.batch_normalization(h)
                h = tf.nn.relu(h)

            prev_dim = h.get_shape()[1]

        with tf.variable_scope('layer%d'%self.n_layer):
            w = tf.get_variable('weight', shape=[h.get_shape()[1], self.dims[-1]],
                                trainable=True,
                                initializer=tf.initializers.truncated_normal(stddev=0.01),
                                dtype=self.dtype)

            self.outputs = tf.matmul(h, w)

        self.preds = tf.gather_nd(self.outputs, inputs.indices)
        pref_diff_zero = tf.reduce_sum(tf.square(self.outputs)) - tf.reduce_sum(tf.square(self.preds))
        pref_diff_ones = tf.reduce_sum(tf.square(self.preds - 1)) * 10000

        self.loss = tf.add_n([pref_diff_ones, pref_diff_zero]) / (self.height * self.width)
        self.loss = tf.identity(self.loss, name='loss')

        all_var = [var for var in tf.trainable_variables() ]

        l2_losses = []
        for var in all_var:
            if var.op.name.find('weight') >= 0 or var.op.name.find('sides') >= 0:
                l2_losses.append(tf.nn.l2_loss(var))

        self.loss = tf.add(self.loss, 2 * self.l2_lambda * tf.reduce_sum(l2_losses), name='total_loss')

        # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # self.loss = tf.add_n([self.loss] + reg_losses, name='total_loss')

    def optimization(self):
        opt = tf.train.AdamOptimizer(self.lr_init)
        train_op = opt.minimize(self.loss, global_step=tf.train.get_global_step(),
                                             name='step_update')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group(self.preload_op, self.gpucopy_op, train_op, update_ops)


    def _BAE_model_fn(self, features, labels, mode, params):
        self.height        = params['height']
        self.width         = params['width']

        # self.lr = tf.train.piecewise_constant(
        #     tf.train.get_global_step(),
        #     [40000, 80000],
        #     [self.init_lr, 0.1*self.init_lr, 0.001*self.init_lr],
        #     name='learning_rate')

        # TODO: Better way instead of tf.identity

        indices       = tf.identity(features['pref'].indices)
        values        = tf.identity(features['pref'].values)
        dense_shape   = tf.identity(features['pref'].dense_shape)

        indices_s     = tf.identity(features['sides'].indices)
        values_s      = tf.identity(features['sides'].values)
        dense_shape_s = tf.identity(features['sides'].dense_shape)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if is_training:
            with tf.device('/cpu:0'):
                # Stage inputs on the host
                self.preload_op, (indices, values, dense_shape, indices_s, values_s, dense_shape_s) \
                    = _stage([indices, values, dense_shape, indices_s, values_s, dense_shape_s])
            with tf.device(self.device):
                # Stage inputs to the device
                self.gpucopy_op, (indices, values, dense_shape, indices_s, values_s, dense_shape_s)\
                    = _stage([indices, values, dense_shape, indices_s, values_s, dense_shape_s])

        inputs = tf.SparseTensor(indices, values, dense_shape)
        sides  = tf.SparseTensor(indices_s, values_s, dense_shape_s)

        with tf.device(self.device):
            inputs = tf.cast(inputs, self.dtype)
            sides  = tf.cast(sides, self.dtype)

            self.builder(inputs, sides)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'preds': self.outputs,
                'ratingTest': tf.sparse.to_dense(features['labels'])
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        with tf.device(None):
            if mode == tf.estimator.ModeKeys.EVAL:
                labels = tf.cast(labels, self.dtype)
                # preds = self.outputs
                # recall = tf.metrics.recall_at_k(
                #     labels=tf.cast(labels, tf.int64), predictions=preds, k=100)
                rmse_tr = tf.metrics.mean_squared_error(
                    labels=inputs.values, predictions=tf.gather_nd(self.outputs, inputs.indices))

                rmse_te = tf.metrics.mean_squared_error(
                    labels=labels.values, predictions=tf.gather_nd(self.outputs, labels.indices))

                tf.summary.scalar('rmse_tr', rmse_tr[1])
                tf.summary.scalar('rmse_te', rmse_te[1])

                metrics = {'rmse_tr': rmse_tr,
                           'rmse_te': rmse_te}

                return tf.estimator.EstimatorSpec(
                    mode, loss=self.loss, eval_metric_ops=metrics)
        assert( mode == tf.estimator.ModeKeys.TRAIN)
        self.optimization()

        return tf.estimator.EstimatorSpec(mode, loss=self.loss, train_op=self.train_op)

    def destroy_graph(self):
        tf.reset_default_graph()

