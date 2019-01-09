import tensorflow as tf
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
        self.batch_size    = params['batch_size']
        self.init_lr       = params['lr']
        self.l2_lambda     = params['l2_lambda']
        self.rank          = params['rank']
        self.eps           = params['eps']
        self.device        = params['device']
        self.log_dir       = params['log_dir']
        self.prefetch_size = params['prefetch_size']

        self.dtype = tf.float16 if params['precision'] == 'fp16' else tf.float32
        self.n_layer = len(self.dims) - 1

    def builder(self, inputs):
        w_init = tf.contrib.layers.xavier_initializer()
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

            if i == 1 and self.n_layer != 2:
                h = tf.sparse.matmul(inputs, w) + b
                # h = tf.layers.batch_normalization(h)
                h = tf.nn.relu(h)

            elif self.n_layer == 2:
                h = tf.sparse.matmul(inputs, w) + b
                # h = tf.layers.batch_normalization(h)
                h = tf.nn.sigmoid(h)

            elif i == (self.n_layer-1):
                h = tf.matmul(h ,w) + b
                # h = tf.layers.batch_normalization(h)
                h = tf.nn.sigmoid(h)

            else:
                h = tf.matmul(h, w) + b
                # h = tf.layers.batch_normalization(h)
                h = tf.nn.relu(h)

            prev_dim = h.get_shape()[1]

        with tf.variable_scope('layer%d'%self.n_layer):
            w = tf.get_variable('weight', shape=[h.get_shape()[1], self.dims[-1]],
                                trainable=True,
                                initializer=w_init,
                                dtype=self.dtype)

        self.outputs = tf.matmul(h, w)

        self.preds = tf.gather_nd(self.outputs, inputs.indices)
        self.loss = tf.losses.mean_squared_error(labels=inputs.values, predictions=self.preds)
        self.loss = tf.identity(self.loss, name='loss')

        all_var = [var for var in tf.trainable_variables() ]

        l2_losses = []
        for var in all_var:
            if var.op.name.find('weight') >= 0:
                l2_losses.append(tf.nn.l2_loss(var))

        self.loss = tf.add(self.loss, self.l2_lambda * tf.reduce_sum(l2_losses), name='total_loss')
        # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # self.loss = tf.add_n([self.loss] + reg_losses, name='total_loss')

    def optimization(self):
        opt = tf.train.AdamOptimizer(self.lr)
        train_op = opt.minimize(self.loss, global_step=tf.train.get_global_step(),
                                             name='step_update')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group(self.preload_op, self.gpucopy_op, train_op, update_ops)


    def _BAE_model_fn(self, features, labels, mode, params):

        self.lr = tf.train.piecewise_constant(
            tf.train.get_global_step(),
            [40000, 80000],
            [self.init_lr, 0.1*self.init_lr, 0.001*self.init_lr],
            name='learning_rate')

        # TODO: Better way instead of tf.identity

        indices = tf.identity(features.indices)
        values = tf.identity(features.values)
        dense_shape = tf.identity(features.dense_shape)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if is_training:
            with tf.device('/cpu:0'):
                # Stage inputs on the host
                self.preload_op, (indices, values, dense_shape) = _stage([indices, values, dense_shape])
            with tf.device(self.device):
                # Stage inputs to the device
                self.gpucopy_op, (indices, values, dense_shape) = _stage([indices, values, dense_shape])

        inputs = tf.SparseTensor(indices, values, dense_shape)

        with tf.device(self.device):
            inputs = tf.cast(inputs, self.dtype)
            self.builder(inputs)


        with tf.device(None):
            if mode == tf.estimator.ModeKeys.EVAL:
                labels = tf.cast(labels, self.dtype)
                preds = tf.gather_nd(self.outputs, labels.indices)
                target = labels.values
                rmse_val = tf.metrics.mean_squared_error(
                    labels=target, predictions=preds)
                rmse_train = tf.metrics.mean_squared_error(
                    labels=inputs.values, predictions=self.preds)

                tf.summary.scalar('rmse_val', rmse_val[1])
                tf.summary.scalar('rmse_train', rmse_train[1])

                metrics = {'rmse_val': rmse_val,
                           'rmse_train': rmse_train}
                return tf.estimator.EstimatorSpec(
                    mode, loss=self.loss, eval_metric_ops=metrics)
        assert( mode == tf.estimator.ModeKeys.TRAIN)
        self.optimization()

        return tf.estimator.EstimatorSpec(mode, loss=self.loss, train_op=self.train_op)

    def destroy_graph(self):
        tf.reset_default_graph()

