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
    fold, cidx, begin, end, height, width, n_contents, data, out_dir, mode = inputs
    col_tr, row_tr, pref_tr, \
    col_te, row_te, pref_te, item, feature, contents, Mask = data

    data = Data()
    train_path = os.path.join(out_dir, '%s.%s.fold.%s.tfrecords' % (mode, cidx, fold))
    train_writer = tf.python_io.TFRecordWriter(train_path)
    num_train, num_val = 0, 0

    sparse_tr = csr_matrix((pref_tr, (row_tr, col_tr)), shape=(height, width))
    sparse_te = csr_matrix((pref_te, (row_te, col_te)), shape=(height, width))
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

        if mode == 'warm':
            col_tr, row_tr, val_tr, row_te, col_te, val_te, cont_row_tr, cont_col_tr, cont_val_tr, cont_row_te, cont_col_te, cont_val_te, masks \
                = self._split_train_val_for_warm(row, columns, rating, item, feature, contents)
        else:
            col_tr, row_tr, val_tr, row_te, col_te, val_te, cont_row_tr, cont_col_tr, cont_val_tr, cont_row_te, cont_col_te, cont_val_te, masks \
                = self._split_train_val_for_cold(row, columns, rating, item, feature, contents)

        for i in range(opt.n_folds):
            # For Training set
            pool = Pool(opt.num_workers)

            data = (col_tr[i], row_tr[i], val_tr[i], [], [], [], cont_row_tr[i], cont_col_tr[i], cont_val_tr[i], None)

            try:
                num_data = pool.map_async(parse_data, [
                    (i, cidx, begin, end, self.height_tr, self.width, self.n_item_feature, data, out_dir, 'train')
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

            # For Test set
            pool = Pool(opt.num_workers)

            if mode == 'warm':
                data = (col_tr[i], row_tr[i], val_tr[i], col_te[i], row_te[i], val_te[i], cont_row_te[i], cont_col_te[i], cont_val_te[i], masks[i])
            else:
                data = ([], [], [], col_te[i], row_te[i], val_te[i], cont_row_te[i], cont_col_te[i], cont_val_te[i], masks[i])

            try:
                num_data = pool.map_async(parse_data, [
                    (i, cidx, begin, end, self.height_te, self.width, self.n_item_feature, data, out_dir, 'test')
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

    def _split_train_val_for_warm(self, row, column, rating, item, feature, content):
        pref = coo_matrix((rating, (row, column)), shape=(self.height, self.width))
        divider = np.random.uniform(0, 1, [self.height, self.width])

        row_tr, col_tr, val_tr, cont_row_tr, cont_col_tr, cont_val_tr = [], [], [], [], [], []
        row_te, col_te, val_te, cont_row_te, cont_col_te, cont_val_te = [], [], [], [], [], []
        self.height_tr, self.height_te = [], []
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

            cont_row_tr.append(item)
            cont_col_tr.append(feature)
            cont_val_tr.append(content)

            cont_row_te.append(item)
            cont_col_te.append(feature)
            cont_val_te.append(content)

            masks.append(mask)

            self.height_tr.append(self.height)
            self.height_te.append(self.height)

        return col_tr, row_tr, val_tr, col_te, row_te, val_te, \
               cont_row_tr, cont_col_tr, cont_val_tr, cont_row_te, cont_col_te, cont_val_te, masks

    def _split_train_val_for_cold(self, row, col, pref, item, feature, content):
        pref = coo_matrix((pref, (row, col)), shape=(self.height, self.width))
        cont = coo_matrix((content, (item, feature)), shape=(self.height, self.n_item_feature))

        n_data = np.max(row) + 1
        rand_idx = np.random.permutation(range(n_data))
        n_one_fold = int(n_data * 0.2)

        row_tr, col_tr, val_tr, cont_row_tr, cont_col_tr, cont_val_tr = [], [], [], [], [], []
        row_te, col_te, val_te, cont_row_te, cont_col_te, cont_val_te = [], [], [], [], [], []
        self.height_tr, self.height_te = [], []
        masks = []

        for i in range(opt.n_folds):
            begin = i * n_one_fold
            end   = np.minimum((i+1)*n_one_fold, n_data)

            test_idx  = rand_idx[begin:end]
            train_idx = np.setdiff1d(range(n_data), test_idx)
            self.height_tr.append(len(train_idx))
            self.height_te.append(len(test_idx))

            train   = pref[train_idx, :]
            val     = pref[test_idx, :]
            cont_tr = cont[train_idx, :]
            cont_te = cont[test_idx, :]

            row_tr.append(train.nonzero()[0])
            col_tr.append(train.nonzero()[1])
            val_tr.append(train.toarray()[train.nonzero()])

            row_te.append(val.nonzero()[0])
            col_te.append(val.nonzero()[1])
            val_te.append(val.toarray()[val.nonzero()])

            cont_row_tr.append(cont_tr.nonzero()[0])
            cont_col_tr.append(cont_tr.nonzero()[1])
            cont_val_tr.append(cont_tr.toarray()[train.nonzero()])

            cont_row_te.append(cont_te.nonzero()[0])
            cont_col_te.append(cont_te.nonzero()[1])
            cont_val_te.append(cont_te.toarray()[train.nonzero()])

            masks.append(np.ones((self.height_te, self.width)))

        return col_tr, row_tr, val_tr, col_te, row_te, val_te, \
               cont_row_tr, cont_col_tr, cont_val_tr, cont_row_te, cont_col_te, cont_val_te, masks


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