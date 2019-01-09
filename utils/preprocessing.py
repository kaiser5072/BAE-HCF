import tensorflow as tf
import os

def _deserialize_data_record(record, mode):
    if mode == 'train':
        feature_map = {
            'column'    : tf.FixedLenFeature([], tf.string, ''),
            'value'     : tf.FixedLenFeature([], tf.string, ''),
            'feature_t' : tf.FixedLenFeature([], tf.string, ''),
            'contents_t': tf.FixedLenFeature([], tf.string, '')
        }
        with tf.name_scope('deserialize_image_record'):
            obj = tf.parse_single_example(record, feature_map)

            item     = obj['column']
            value    = obj['value']
            feature  = obj['feature_t']
            contents = obj['contents_t']

            return item, value, feature, contents, None, None, None
    else:
        feature_map = {
            'column'   : tf.FixedLenFeature([], tf.string, ''),
            'value'    : tf.FixedLenFeature([], tf.string, ''),
            'column_v' : tf.FixedLenFeature([], tf.string, ''),
            'value_v'  : tf.FixedLenFeature([], tf.string, ''),
            'feature_t': tf.FixedLenFeature([], tf.string, ''),
            'contents_t': tf.FixedLenFeature([], tf.string, ''),
            'mask'      : tf.FixedLenFeature([], tf.string, '')
        }

        with tf.name_scope('deserialize_image_record'):
            obj = tf.parse_single_example(record, feature_map)

            item, value       = obj['column'], obj['value']
            item_v, value_v   = obj['column_v'], obj['value_v']
            feature, contents = obj['feature_t'], obj['contents_t']
            mask = obj['mask']

            return item, value, feature, contents, item_v, value_v, mask

def _parse_and_preprocess_record(record, width, mode):

    item, value, feature, content, item_v, value_v, mask = _deserialize_data_record(record, mode)

    item  = tf.decode_raw(item, out_type=tf.int32)
    value = tf.decode_raw(value, out_type=tf.float32)
    feature = tf.decode_raw(feature, out_type=tf.int32)
    content = tf.decode_raw(content, out_type=tf.int8)

    item = tf.cast(item, tf.int64)
    feature = tf.cast(feature, tf.int64)

    inputs = tf.SparseTensor(tf.reshape(item, [-1, 1]), value, [width])
    sides = tf.SparseTensor(tf.reshape(feature, [-1, 1]), content, [8000])

    if mode == 'train':
        inputs = {'pref': inputs, 'sides': sides}
        return inputs

    else:
        item_v = tf.decode_raw(item_v, out_type=tf.int32)
        value_v = tf.decode_raw(value_v, out_type=tf.float32)
        mask    = tf.decode_raw(mask, out_type=tf.int8)

        item_v = tf.cast(item_v, tf.int64)

        labels = tf.SparseTensor(tf.reshape(item_v, [-1, 1]), value_v, [width])

        inputs = {'pref': inputs, 'sides': sides, 'mask': mask, 'labels': labels}

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