import tensorflow as tf
import numpy as np
import h5py
import cPickle
import shutil
import fire
import tqdm
import os

from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from multiprocessing import Pool
from utils import get_logger, Option
opt = Option('./config.json')


def parse_data(inputs):
    fold, cidx, begin, end, height, width, n_contents, data, item, feature, contents, out_dir = inputs
    col_tr, row_tr, pref_tr, \
    col_te, row_te, pref_te, Mask = data

    data = Data()
    train_path = os.path.join(out_dir, 'train.%s.fold.%s.tfrecords' % (cidx, fold))
    train_writer = tf.python_io.TFRecordWriter(train_path)
    num_train, num_val = 0, 0

    sparse_tr = csr_matrix((pref_tr, (row_tr, col_tr)), shape=(height, width))
    print('a')
    sparse_te = csr_matrix((pref_te, (row_te, col_te)), shape=(height, width))
    print('b')
    sparse_co = csr_matrix((contents, (item, feature)), shape=(height, n_contents))

    with tqdm.tqdm(total=end-begin) as pbar:
        for column, value, column_v, value_v, feature_t, contents_t, mask in data.generate(sparse_tr,
                                                                                           sparse_te,
                                                                                           sparse_co,
                                                                                           Mask,
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

    def generate(self, train, val, contents, mask, begin, end):
        for i in range(begin, end):
            col_tr  = train[i].indices
            pref_tr = train[i].data

            col_te  = val[i].indices
            pref_te = val[i].data

            feature_t  = contents[i].indices
            contents_t = contents[i].data

            if col_tr.size == 0 and pref_tr.size == 0 and feature_t.size == 0 and contents_t.size == 0:
                continue

            yield col_tr, pref_tr, col_te, pref_te, feature_t, contents_t, mask[i, :]

    def make_db(self, data_dir, out_dir, mode):
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        os.makedirs(out_dir)

        row, columns, rating, item, feature, contents = self.load_data(data_dir, 'item')
        chunk_offsets = self._split_data(row, opt.chunk_size)
        num_chunks = len(chunk_offsets)
        self.logger.info('split data into %d chunks' % (num_chunks))

        col_tr, row_tr, val_tr, row_te, col_te, val_te, masks = self._split_train_val(row, columns, rating, mode)


        for i in range(opt.n_folds):
            pool = Pool(opt.num_workers)
            data = (col_tr[i], row_tr[i], val_tr[i], row_te[i], col_te[i], val_te[i], masks[i])
            try:
                num_data = pool.map_async(parse_data, [(i, cidx, begin, end, self.height, self.width, self.n_item_feature, data, item, feature, contents, out_dir)
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

            meta_fout = open(os.path.join(out_dir, 'meta%s'%i), 'w')
            meta = {'num_train' : num_train,
                    'num_val'   : num_val,
                    'height'    : self.height,
                    'width'     : self.width}
            meta_fout.write(cPickle.dumps(meta, 2))
            meta_fout.close()

            self.logger.info('size of training set: %s' % num_train)
            self.logger.info('size of validation set: %s' % num_val)
            self.logger.info('height: %s' % self.height)
            self.logger.info('width: %s' % self.width)

    def _split_train_val(self, row, column, rating, mode):
        pref = coo_matrix((rating, (row, column)), shape=(self.height, self.width))
        divider = np.random.uniform(0, 1, [self.height, self.width])

        row_tr, col_tr, val_tr = [], [], []
        row_te, col_te, val_te = [], [], []
        masks = []
        for i in range(opt.n_folds):
            divider[divider < (i+1)*0.2] = 5 - i
            mask = np.zeros_like(divider)
            mask[divider == 5 - i] = 1

            train = pref.multiply(1 - mask)
            val   = pref.multiply(mask)

            row_tr.append(train.nonzero()[0])
            col_tr.append(train.nonzero()[1])
            val_tr.append(train.toarray()[train.nonzero()])

            row_te.append(val.nonzero()[0])
            col_te.append(val.nonzero()[1])
            val_te.append(val.toarray()[val.nonzero()])

            masks.append(mask)

        return col_tr, row_tr, val_tr, row_te, col_te, val_te, masks


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