import tensorflow as tf
import os

def _deserialize_data_record(record, mode):
    if mode == 'train':
        feature_map = {
            'column'  : tf.FixedLenFeature([], tf.string, ''),
            'value'   : tf.FixedLenFeature([], tf.string, '')}

        with tf.name_scope('deserialize_image_record'):
            obj = tf.parse_single_example(record, feature_map)

            item  = obj['column']
            value = obj['value']

            return item, value, None, None
    else:
        feature_map = {
            'column'   : tf.FixedLenFeature([], tf.string, ''),
            'value'    : tf.FixedLenFeature([], tf.string, ''),
            'column_v' : tf.FixedLenFeature([], tf.string, ''),
            'value_v'  : tf.FixedLenFeature([], tf.string, '')}

        with tf.name_scope('deserialize_image_record'):
            obj = tf.parse_single_example(record, feature_map)

            item, value     = obj['column'], obj['value']
            item_v, value_v = obj['column_v'], obj['value_v']

            return item, value, item_v, value_v

def _parse_and_preprocess_record(record, width, mode):

    item, value, item_v, value_v = _deserialize_data_record(record, mode)

    item  = tf.decode_raw(item, out_type=tf.int32)
    value = tf.decode_raw(value, out_type=tf.float32)

    item = tf.cast(item, tf.int64)

    inputs = tf.SparseTensor(tf.reshape(item, [-1, 1]), value, [width])

    if mode == 'train':
        return inputs

    else:
        item_v = tf.decode_raw(item_v, out_type=tf.int32)
        value_v = tf.decode_raw(value_v, out_type=tf.float32)

        item_v = tf.cast(item_v, tf.int64)

        labels = tf.SparseTensor(tf.reshape(item_v, [-1, 1]), value_v, [width])

        return inputs, labels

def data_set(data_dir, batch_size, prefetch_size, width, mode):


    data_path = os.path.join(data_dir, 'train.*.tfrecords')


    filenames = tf.data.Dataset.list_files(data_path)

    ds = filenames.apply(
        tf.data.experimental.parallel_interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=4))

    # if training:
    #     ds = ds.shuffle(10000)

    if mode == 'train':
        ds = ds.shuffle(100000)
        ds = ds.repeat()

    preproc_func = lambda record: _parse_and_preprocess_record(record, width, mode)

    ds = ds.apply(tf.data.experimental.map_and_batch(
        map_func=preproc_func,
        batch_size=batch_size,
        num_parallel_calls=32))

    ds = ds.prefetch(prefetch_size)

    return ds