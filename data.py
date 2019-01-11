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
    col_t, row_t, rating_t = DATA

    data = Data()
    train_path = os.path.join(out_dir, 'train.%s.tfrecords' % cidx)
    train_writer = tf.python_io.TFRecordWriter(train_path)
    num_train, num_val = 0, 0
    with tqdm.tqdm(total=end-begin) as pbar:
        for column, value, feature_t, contents_t, in data.generate(row_t,
                                                                   col_t,
                                                                   rating_t,
                                                                   item,
                                                                   feature,
                                                                   contents,
                                                                   begin, end):
            num_train += len(value)
            value      = value.astype(np.int8)
            contents_t = contents_t.astype(np.float32)

            example_train = tf.train.Example(features=tf.train.Features(feature={
                'column'    : data._byte_feature(column.tostring()),
                'value'     : data._byte_feature(value.tostring()),
                'feature_t' : data._byte_feature(feature_t.tostring()),
                'contents_t': data._byte_feature(contents_t.tostring())
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

    def generate(self, row_t, col_t, rating_t, item, feature, contents, begin, end):


        for i in range(begin, end):
            # train = csr_matrix((rating_train, (row_train, column_train)), shape=(self.height, self.width))
            # val   = csr_matrix((rating_val, (row_val, column_val)), shape=(self.height, self.width))
            train_index = (row_t == i)
            column_t  = col_t[train_index]
            value_t = rating_t[train_index]

            contents_index = (item == i)
            feature_t = feature[contents_index]
            contents_t = contents[contents_index]

            if column_t.size == 0 and value_t.size == 0 and feature_t.size == 0 and contents_t.size == 0:
                continue
            # column_t = train[i, :].indices
            # value_t  = train[i, :].data
            #
            # column_v = val[i, :].indices
            # value_v  = val[i, :].data

            yield column_t, value_t, feature_t, contents_t


    def make_db(self, data_dir, out_dir, train_ratio):
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        os.makedirs(out_dir)

        row, columns, rating, item, feature, contents = self.load_data(data_dir, 'item')
        chunk_offsets = self._split_data(row, opt.chunk_size)
        num_chunks = len(chunk_offsets)
        self.logger.info('split data into %d chunks' % (num_chunks))

        # data = self._split_train_val(row, columns, rating, train_ratio)
        data = (row, columns, rating)

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
                val.nonzero()[1], val.nonzero()[0], val.toarray()[val.nonzero()], mask)


    def _split_data(self, row, chunk_size):
        total = np.max(row) + 1
        chunks = [(i, min(i + chunk_size, total))
                  for i in range(0, total, chunk_size)]

        return chunks

    def _byte_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def build_data(self, data_dir):
        ''' Remove duplicate interactions and collapse remaining interactions
        into a single binary matrix in the form of a sparse matrix'''
        # Training set
        data_path = os.path.join(data_dir, 'recsys2017.pub/eval/warm/train.csv')
        pref = np.loadtxt(data_path, dtype='int32, int32, float32',
                          delimiter=',',
                          usecols=(0, 1, 2),
                          unpack=True)
        pref[2] = np.ones(np.shape(pref[2]))

        uniq = list(set(zip(pref[0], pref[1], pref[2])))
        uniq = uniq.sort()

        pref = zip(*uniq)

        with open(os.path.join(data_dir, 'recsys2017.pub/eval/item_features_0based.txt', 'r')) as f:
            item_feature, i = [], 0
            for line in f:
                for item in line.split()[1:]:
                    item_feature.append([i, int(item.split(':')[0]), float(item.split(':')[1])])
                i = i + 1

        item_feature = item_feature.sort()
        item_feature = zip(*item_feature)


if __name__ == '__main__':
    data = Data()
    fire.Fire({'make_db': data.make_db})