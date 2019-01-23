import tensorflow as tf
import numpy as np
import h5py
import cPickle
import shutil
import fire
import tqdm
import os

from scipy.sparse import csr_matrix
from multiprocessing import Pool
from utils import get_logger, Option
opt = Option('./config.json')


def parse_data(inputs):
    cidx, begin, end, data, height, width, n_features, div, out_dir = inputs
    row_tr, col_tr, pref_tr, item, feature, contents, row_te, col_te, pref_te = data

    data = Data()
    train_path = os.path.join(out_dir, '%s.%s.tfrecords' % (div, cidx))
    train_writer = tf.python_io.TFRecordWriter(train_path)
    num_train, num_val = 0, 0
    sparse_tr = csr_matrix((pref_tr, (row_tr, col_tr)), shape=(height, width))
    sparse_co = csr_matrix((contents, (item, feature)), shape=(height, n_features))
    with tqdm.tqdm(total=end-begin) as pbar:
        for column, value, feature_t, contents_t, Col_te, Pref_te in data.generate(sparse_tr,
                                                                                   sparse_co,
                                                                                   row_te,
                                                                                   col_te,
                                                                                   pref_te,
                                                                                   begin, end):
            num_train += len(value)
            value      = value.astype(np.int8)
            contents_t = contents_t.astype(np.float32)

            if div == 'train':
                example_train = tf.train.Example(features=tf.train.Features(feature={
                    'column'    : data._byte_feature(column.tostring()),
                    'value'     : data._byte_feature(value.tostring()),
                    'feature_t' : data._byte_feature(feature_t.tostring()),
                    'contents_t': data._byte_feature(contents_t.tostring())
                }))
            else:
                Pref_te = Pref_te.astype(np.int8)
                example_train = tf.train.Example(features=tf.train.Features(feature={
                    'column'    : data._byte_feature(column.tostring()),
                    'value'     : data._byte_feature(value.tostring()),
                    'feature_t' : data._byte_feature(feature_t.tostring()),
                    'contents_t': data._byte_feature(contents_t.tostring()),
                    'column_te' : data._byte_feature(Col_te.tostring()),
                    'value_te'  : data._byte_feature(Pref_te.tostring())
                }))
            train_writer.write(example_train.SerializeToString())
            pbar.update(1)

    train_writer.close()

    return num_train, num_val

class Data(object):
    def __init__(self):
        self.logger = get_logger('data')

    def load_data(self, data_dir, div, AE_TYPE='item'):
        data = h5py.File(data_dir, 'r')

        if AE_TYPE == 'item':
            row      = data['pref']['item'][:]
            column   = data['pref']['user'][:]
            item     = data['item-contents']['item'][:]
            feature  = data['item-contents']['feature'][:]
            contents = data['item-contents']['value'][:]

        else:
            row    = data['pref']['user'][:]
            column = data['pref']['item'][:]
            item     = data['user-contents']['user'][:]
            feature  = data['user-contents']['feature'][:]
            contents = data['user-contents']['value'][:]

        pref = data['pref']['value'][:]

        self.height = np.max(item) + 1
        self.width  = np.max(column) + 1
        self.n_contents = np.max(feature) + 1

        if div == 'train':
            return (row, column, pref, item, feature, contents, None, None, None)

        else:
            if AE_TYPE == 'item':
                row_te = data['labels']['item'][:]
                col_te = data['labels']['user'][:]
            else:
                row_te = data['labels']['user'][:]
                col_te = data['labels']['item'][:]

            pref_te = data['labels']['value'][:]

            return (row, column, pref, item, feature, contents, row_te, col_te, pref_te)

    def generate(self, train, contents, row_te, col_te, pref_te, begin, end):
        for i in range(begin, end):
            column_t    = train[i].indices
            value_t     = train[i].data

            feature_t      = contents[i].indices
            contents_t     = contents[i].data

            if row_te is not None:
                index_te = (row_te == i)
                Col_te   = col_te[index_te]
                Pref_te  = pref_te[index_te]

            else:
                Col_te = None
                Pref_te = None

            if column_t.size == 0 and value_t.size == 0 and feature_t.size == 0 and contents_t.size == 0:
                continue

            yield column_t, value_t, feature_t, contents_t, Col_te, Pref_te


    def make_db(self, data_dir, out_dir, div, mode):
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        os.makedirs(out_dir)

        data = self.load_data(data_dir, div, mode)
        chunk_offsets = self._split_data(opt.chunk_size)
        num_chunks = len(chunk_offsets)
        self.logger.info('split data into %d chunks' % (num_chunks))

        # data = self._split_train_val(row, columns, rating, train_ratio)
        pool = Pool(opt.num_workers)

        try:
            num_data = pool.map_async(parse_data, [(cidx, begin, end, data, self.height, self.width, self.n_contents, div, out_dir)
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
        meta = {'num_train' : num_train,
                'num_val'   : num_val,
                'height'    : self.height,
                'width'     : self.width,
                'n_features': self.n_contents}
        meta_fout.write(cPickle.dumps(meta, 2))
        meta_fout.close()

        self.logger.info('size of training set: %s' % num_train)
        self.logger.info('size of validation set: %s' % num_val)
        self.logger.info('height: %s' % self.height)
        self.logger.info('width: %s' % self.width)
        self.logger.info('The number of content features: %s' % self.n_contents)

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


    def _split_data(self, chunk_size):
        total = self.height
        chunks = [(i, min(i + chunk_size, total))
                  for i in range(0, total, chunk_size)]

        return chunks

    def _byte_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    #TODO: Require Re-Factoring
    def build_data(self, data_dir, mode):
        ''' Remove duplicate interactions and collapse remaining interactions
        into a single binary matrix in the form of a sparse matrix'''
        # Training set
        data_path = os.path.join(data_dir, 'Input/warm/train.csv')
        user, item, value = self.load_preference_matrix(data_path)

        value = np.ones(np.shape(value))

        uni = list(set(zip(user, item, value)))
        uni.sort()

        user, item, value = zip(*uni)

        user_list = np.unique(user)
        user_dict = dict()
        for i, j in enumerate(user_list):
            user_dict[j] = i

        item_list = np.unique(item)
        item_dict = dict()
        for i, j in enumerate(item_list):
            item_dict[j] = i

        user = [user_dict[i] for i in user]
        item = [item_dict[i] for i in item]

        self.logger.info('Loaded a preference matrix')

        item_content_dir = os.path.join(data_dir, 'Input/warm/item_features_0based.txt')
        item_feature = self.load_content_information(item_content_dir)
        item_feature.sort()

        item_content_ids, item_feature_ids, item_value = zip(*item_feature)
        item_content_ids, item_feature_ids, item_value \
            = np.asarray(item_content_ids), np.asarray(item_feature_ids), np.asarray(item_value)

        content_item = [(i, item_dict[j]) for i, j in enumerate(item_content_ids) if j in item_dict]
        content_item = np.asarray(content_item)
        content_ind  = content_item[:, 0]
        content_item = content_item[:, 1]

        item_feature       = item_feature_ids[content_ind]
        content_item_value = item_value[content_ind]

        user_content_dir = os.path.join(data_dir, 'Input/warm/user_features_0based.txt')
        user_feature = self.load_content_information(user_content_dir)
        user_feature.sort()

        user_content_ids, user_feature_ids, user_value = zip(*user_feature)
        user_content_ids, user_feature_ids, user_value = \
            np.asarray(user_content_ids), np.asarray(user_feature_ids), np.asarray(user_value)

        content_user = [(i, user_dict[j]) for i, j in enumerate(user_content_ids) if j in user_dict]
        content_user = np.asarray(content_user)
        content_ind  = content_user[:, 0]
        content_user = content_user[:, 1]

        user_feature       = user_feature_ids[content_ind]
        content_user_value = user_value[content_ind]

        with h5py.File('./Input/recsys2017_warm.h5py', 'w') as data:
            # Preference matrix
            pref = data.create_group('pref')
            users  = pref.create_dataset('user', np.shape(user), 'i')
            items  = pref.create_dataset('item', np.shape(item), 'i')
            values = pref.create_dataset('value', np.shape(value), 'i')
            users[:], items[:], values[:]  = user, item, value

            # Item Contents
            item_con      = data.create_group('item-contents')
            content_items = item_con.create_dataset('item', np.shape(content_item), 'i')
            item_features = item_con.create_dataset('feature', np.shape(item_feature), 'i')
            contents      = item_con.create_dataset('value', np.shape(content_item_value), 'f')
            content_items[:], item_features[:], contents[:] \
                = content_item, item_feature, content_item_value

            # User Contents
            user_con = data.create_group('user-contents')

            content_users       = user_con.create_dataset('user', np.shape(content_user), 'i')
            user_features       = user_con.create_dataset('feature', np.shape(user_feature), 'i')
            contents_user_value = user_con.create_dataset('value', np.shape(content_user_value), 'f')
            content_users[:], user_features[:], contents_user_value[:] \
                = content_user, user_feature, content_user_value

        self.logger.info('Loading a preference matrix for test')

        '''Test data for Warm start'''
        test_pref_dir = os.path.join(data_dir, 'Input/warm/test_warm.csv')
        user_te, item_te, value_te = self.load_preference_matrix(test_pref_dir)
        value_te = np.ones_like(value_te)

        uni = list(set(zip(user_te, item_te, value_te)))
        uni.sort()

        user_te, item_te, value_te = zip(*uni)

        user_te = [user_dict[i] for i in user_te]
        item_te = [item_dict[i] for i in item_te]

        train_user, train_item, train_value = np.asarray(user), np.asarray(item), np.asarray(value)

        if mode == 'item':
            test_item_list = np.unique(item_te)
            test_item_dict = dict()
            for i, j in enumerate(test_item_list):
                test_item_dict[j] = i

            item_te = [test_item_dict[i] for i in item_te]

            train_item = [(i, test_item_dict[j]) for i, j in enumerate(train_item) if j in test_item_dict]
            train_item = np.asarray(train_item)
            train_ind   = train_item[:, 0]
            train_item  = train_item[:, 1]
            train_user  = train_user[train_ind]
            train_value = train_value[train_ind]

            train_content_item = [(i, test_item_dict[j]) for i, j in enumerate(content_item) if j in test_item_dict]
            train_content_item = np.asarray(train_content_item)
            train_content_ind = train_content_item[:, 0]
            train_content_ui = train_content_item[:, 1]
            train_feature = item_feature[train_content_ind]
            train_content = content_item_value[train_content_ind]

        else:
            test_user_list = np.unique(user_te)
            test_user_dict = dict()
            for i, j in enumerate(test_user_list):
                test_user_dict[j] = i

            user_te = [test_user_dict[i] for i in user_te]

            train_user  = [(i, test_user_dict[j]) for i, j in enumerate(train_user) if j in test_user_dict]
            train_user  = np.asarray(train_user)
            train_ind   = train_user[:, 0]
            train_user  = train_user[:, 1]
            train_item  = train_item[train_ind]
            train_value = train_value[train_ind]

            train_content_user = [(i, test_user_dict[j]) for i, j in enumerate(content_user) if j in test_user_dict]
            train_content_user = np.asarray(train_content_user)
            train_content_ind = train_content_user[:, 0]
            train_content_ui = train_content_user[:, 1]
            train_feature = user_feature[train_content_ind]
            train_content = content_user_value[train_content_ind]

        data_path = os.path.join(data_dir, 'Input/warm/test_warm_item_ids.csv')
        test_warm_item_ids  = np.loadtxt(data_path, dtype='int32')
        test_warm_item_list = [item_dict[i] for i in test_warm_item_ids]

        warm_index = h5py.File('warm_index.h5py', 'w')
        test_warm_item_lists = warm_index.create_dataset('idx', np.shape(test_warm_item_list), 'i')
        test_warm_item_lists[:] = test_warm_item_list

        with h5py.File('./Input/test_warm.h5py', 'w') as data:
            pref = data.create_group('pref')
            users = pref.create_dataset('user', np.shape(train_user), 'i')
            users[:] = train_user
            items = pref.create_dataset('item', np.shape(train_item), 'i')
            items[:] = train_item
            values = pref.create_dataset('value', np.shape(train_value), 'i')
            values[:] = train_value

            if mode == 'item':
                feature_con      = data.create_group('item-contents')
                content_items    = feature_con.create_dataset('item', np.shape(train_content_ui), 'i')
                content_items[:] = train_content_ui
            else:
                feature_con      = data.create_group('user-contents')
                content_users    = feature_con.create_dataset('user', np.shape(train_content_ui), 'i')
                content_users[:] = train_content_ui

            features = feature_con.create_dataset('feature', np.shape(train_feature), 'i')
            features[:] = train_feature

            contents = feature_con.create_dataset('value', np.shape(train_content), 'f')
            contents[:] = train_content

            labels = data.create_group('labels')

            test_users = labels.create_dataset('user', np.shape(user_te), 'i')
            test_users[:] = user_te

            test_items = labels.create_dataset('item', np.shape(item_te), 'i')
            test_items[:] = item_te

            test_values = labels.create_dataset('value', np.shape(value_te), 'i')
            test_values[:] = value_te


        ''' Test data for user cold start '''
        test_pref_usr_cold_dir = os.path.join(data_dir, 'Input/warm/test_cold_user.csv')
        user_te_usr_cold, item_te_usr_cold, value_te_usr_cold\
            = self.load_preference_matrix(test_pref_usr_cold_dir)
        value_te_usr_cold = np.ones_like(value_te_usr_cold)

        uni = list(set(zip(user_te_usr_cold, item_te_usr_cold, value_te_usr_cold)))
        uni.sort()

        user_te_usr_cold, item_te_usr_cold, value_te_usr_cold = zip(*uni)

        user_cold_list = np.unique(user_te_usr_cold)
        user_cold_dict = dict()
        for i, j in enumerate(user_cold_list):
            user_cold_dict[j] = i

        user_te_usr_cold = [user_cold_dict[i] for i in user_te_usr_cold]
        item_te_usr_cold = [item_dict[i] for i in item_te_usr_cold]

        user_tr_usr_cold, item_tr_usr_cold, value_tr_usr_cold = [], [], []

        user_content_usr_cold = [(i, user_cold_dict[j]) for i, j in enumerate(user_content_ids) if j in user_cold_dict]
        user_content_usr_cold = np.asarray(user_content_usr_cold)
        user_cold_ids = user_content_usr_cold[:, 0]
        user_content_usr_cold = user_content_usr_cold[:, 1]
        user_feature_usr_cold = user_feature_ids[user_cold_ids]
        user_value_usr_cold   = user_value[user_cold_ids]

        with h5py.File('./Input/test_user_cold.h5py', 'w') as data:
            pref = data.create_group('pref')
            users  = pref.create_dataset('user', np.shape(train_user), 'i')
            items  = pref.create_dataset('item', np.shape(train_item), 'i')
            values = pref.create_dataset('value', np.shape(train_value), 'i')
            users[:]  = user_tr_usr_cold
            items[:]  = item_tr_usr_cold
            values[:] = value_tr_usr_cold

            feature_con      = data.create_group('user-contents')
            content_users    = feature_con.create_dataset('user', np.shape(train_content_ui), 'i')
            features         = feature_con.create_dataset('feature', np.shape(train_feature), 'i')
            contents         = feature_con.create_dataset('value', np.shape(train_content), 'f')
            content_users[:] = user_content_usr_cold
            features[:]      = user_feature_usr_cold
            contents[:]      = user_value_usr_cold

            labels = data.create_group('labels')
            test_users  = labels.create_dataset('user', np.shape(user_te), 'i')
            test_items  = labels.create_dataset('item', np.shape(item_te), 'i')
            test_values = labels.create_dataset('value', np.shape(value_te), 'i')
            test_users[:]  = user_te_usr_cold
            test_items[:]  = item_te_usr_cold
            test_values[:] = value_te_usr_cold

        data_path = os.path.join(data_dir, 'Input/warm/test_cold_user_item_ids.csv')
        test_cold_user_item_ids = np.loadtxt(data_path, dtype='int32')
        test_cold_user_item_list = [item_dict[i] for i in test_cold_user_item_ids]

        user_cold_index = h5py.File('cold_user_index.h5py', 'w')
        test_cold_user_item_lists = user_cold_index.create_dataset('idx', np.shape(test_cold_user_item_list), 'i')
        test_cold_user_item_lists[:] = test_cold_user_item_list

    def load_preference_matrix(self, data_path):
        pref = np.loadtxt(data_path, dtype='int32, int32, float32',
                                     delimiter=',',
                                     usecols=(0, 1, 2),
                                     unpack=True)

        return pref[0], pref[1], pref[2]

    def load_content_information(self, data_path):
        with open(data_path, 'r') as f:
            contents, i = [], 0
            for line in f:
                for content_ids in line.split()[1:]:
                    contents.append([i, int(content_ids.split(':')[0]), float(content_ids.split(':')[1])])
                i = i + 1

        return contents

if __name__ == '__main__':
    data = Data()
    fire.Fire({'make_db': data.make_db,
               'build_data': data.build_data})