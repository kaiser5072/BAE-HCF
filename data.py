import tensorflow as tf
import numpy as np
import h5py
import cPickle
import shutil
import fire
import tqdm
import os

from scipy.sparse import coo_matrix
from multiprocessing import Pool
from utils import get_logger, Option
opt = Option('./config.json')


def parse_data(inputs):
    cidx, begin, end, DATA, item, feature, contents, out_dir = inputs
    col_t, row_t, rating_t, col_v, row_v, rating_v, mask = DATA

    data = Data()
    train_path = os.path.join(out_dir, 'train.%s.tfrecords' % cidx)
    train_writer = tf.python_io.TFRecordWriter(train_path)

    num_train, num_val = 0, 0
    with tqdm.tqdm(total=end-begin) as pbar:
        for column, value, column_v, value_v, feature_t, contents_t, mask in data.generate(row_t,
                                                                                     col_t,
                                                                                     rating_t,
                                                                                     row_v,
                                                                                     col_v,
                                                                                     rating_v,
                                                                                     mask,
                                                                                     item,
                                                                                     feature,
                                                                                     contents,
                                                                                     begin, end):
            num_train += len(value)
            num_val   += len(value_v)
            value      = value.astype(np.float32)
            value_v    = value_v.astype(np.float32)
            contents_t = contents_t.astype(np.int8)
            mask       = mask.astype(np.int8)

            example_train = tf.train.Example(features=tf.train.Features(feature={
                'column'    : data._byte_feature(column.tostring()),
                'value'     : data._byte_feature(value.tostring()),
                'column_v'  : data._byte_feature(column_v.tostring()),
                'value_v'   : data._byte_feature(value_v.tostring()),
                'feature_t' : data._byte_feature(feature_t.tostring()),
                'contents_t': data._byte_feature(contents_t.tostring()),
                'mask'      : data._byte_feature(mask.tostring())
            }))
            train_writer.write(example_train.SerializeToString())
            pbar.update(1)

    train_writer.close()

    return num_train, num_val

class Data(object):
    def __init__(self):
        self.logger = get_logger('data')

    def load_data(self, data_dir, AE_TYPE='item'):
        data = h5py.File(data_dir, 'r')

        if AE_TYPE == 'item':
            row    = data['pref']['item'][:]
            column = data['pref']['user'][:]
        else:
            row    = data['pref']['user'][:]
            column = data['pref']['item'][:]

        pref = data['pref']['value'][:]

        self.height = np.max(row) + 1
        self.width  = np.max(column) + 1

        item     = data['item-contents']['item'][:]
        feature  = data['item-contents']['feature'][:]
        contents = data['item-contents']['value'][:]

        self.n_item_feature = np.max(feature) + 1

        return row, column, pref, item, feature, contents

    def generate(self, row_t, col_t, rating_t, row_v, col_v, rating_v, mask, item, feature, contents, begin, end):


        for i in range(begin, end):
            # train = csr_matrix((rating_train, (row_train, column_train)), shape=(self.height, self.width))
            # val   = csr_matrix((rating_val, (row_val, column_val)), shape=(self.height, self.width))
            train_index = (row_t == i)
            column_t  = col_t[train_index]
            value_t = rating_t[train_index]

            val_index = (row_v == i)
            column_v  = col_v[val_index]
            value_v = rating_v[val_index]

            contents_index = (item == i)
            feature_t = feature[contents_index]
            contents_t = contents[contents_index]
            # column_t = train[i, :].indices
            # value_t  = train[i, :].data
            #
            # column_v = val[i, :].indices
            # value_v  = val[i, :].data
            print(mask[i, :])

            yield column_t, value_t, column_v, value_v, feature_t, contents_t, mask[i, :]


    def make_db(self, data_dir, out_dir, train_ratio):
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        os.makedirs(out_dir)

        row, columns, rating, item, feature, contents = self.load_data(data_dir, 'item')
        chunk_offsets = self._split_data(row, opt.chunk_size)
        num_chunks = len(chunk_offsets)
        self.logger.info('split data into %d chunks' % (num_chunks))

        data = self._split_train_val(row, columns, rating, train_ratio)

        pool = Pool(opt.num_workers)

        try:
            num_data = pool.map_async(parse_data, [(cidx, begin, end, data, item, feature, contents, out_dir)
                                                   for cidx, (begin, end) in enumerate(chunk_offsets)]).get(999999999)
            pool.close()
            pool.join()

        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise

        num_train, num_val = 0, 0
        for train, val in num_data:
            num_train += train
            num_val += val

        meta_fout = open(os.path.join(out_dir, 'meta'), 'w')
        meta = {'num_train': num_train,
                'num_val': num_val,
                'height': self.height,
                'width': self.width}
        meta_fout.write(cPickle.dumps(meta, 2))
        meta_fout.close()

        self.logger.info('size of training set: %s' % num_train)
        self.logger.info('size of validation set: %s' % num_val)
        self.logger.info('height: %s' % self.height)
        self.logger.info('width: %s' % self.width)

    def _split_train_val(self, row, column, rating, train_ratio):
        # val_ratio = int(1 / (1 - train_ratio))
        # val_index = np.random.choice(len(rating), len(rating)//val_ratio, replace=False)
        # val_index = np.sort(val_index)
        # train_index = np.setdiff1d(np.arange(len(rating)), val_index)

        pref = coo_matrix((rating, (row, column)), shape=(self.height, self.width))
        divider = np.random.uniform(0, 1, [self.height, self.width])
        mask = np.zeros_like(divider)
        mask[divider > train_ratio] = 1

        train = pref.multiply(1 - mask)
        val   = pref.multiply(mask)

        return (train.nonzero()[1], train.nonzero()[0], train.toarray()[train.nonzero()],
                val.nonzero()[1], val.nonzero()[0], val.toarray()[train.nonzero()], mask)


    def _split_data(self, row, chunk_size):
        total = np.max(row) + 1
        chunks = [(i, min(i + chunk_size, total))
                  for i in range(0, total, chunk_size)]

        return chunks

    def _byte_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == '__main__':
    data = Data()
    fire.Fire({'make_db': data.make_db})