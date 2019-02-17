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
    cidx, begin, end, data, n_items_height, n_items_width, n_users_height, n_users_width, n_contents_items, n_contents_users, div, out_dir, types = inputs
    item_row_tr, item_col_tr, item_ids, item_feature, item_contents, \
    user_row_tr, user_col_tr, user_ids, user_feature, user_contents, \
    item_pref_tr, user_pref_tr = data

    data = Data()
    train_path = os.path.join(out_dir, '%s.%s.%s.tfrecords' % (types, div, cidx))
    train_writer = tf.python_io.TFRecordWriter(train_path)
    num_train, num_val = 0, 0
    if types == 'item':
        sparse_tr = csr_matrix((item_pref_tr, (item_row_tr, item_col_tr)), shape=(n_items_height, n_items_width))
        sparse_co = csr_matrix((item_contents, (item_ids, item_feature)), shape=(n_items_height, n_contents_items))
    else:
        sparse_tr = csr_matrix((user_pref_tr, (user_row_tr, user_col_tr)), shape=(n_users_height, n_users_width))
        sparse_co = csr_matrix((user_contents, (user_ids, user_feature)), shape=(n_users_height, n_contents_users))

    with tqdm.tqdm(total=end-begin) as pbar:
        for column, value, feature_t, contents_t in data.generate(sparse_tr, sparse_co, begin, end):
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

    def load_data(self, data_dir, div):
        data = h5py.File(data_dir, 'r')


        item_row      = data['item_based']['pref']['item'][:]
        item_col      = data['item_based']['pref']['user'][:]
        item_pref     = data['item_based']['pref']['value'][:]
        item_ids      = data['item_based']['contents']['item'][:]
        item_feature  = data['item_based']['contents']['feature'][:]
        item_contents = data['item_based']['contents']['value'][:]

        user_row      = data['user_based']['pref']['user'][:]
        user_col      = data['user_based']['pref']['item'][:]
        user_pref     = data['user_based']['pref']['value'][:]
        user_ids      = data['user_based']['contents']['user'][:]
        user_feature  = data['user_based']['contents']['feature'][:]
        user_contents = data['user_based']['contents']['value'][:]

        meta_path = os.path.join('./Input', 'meta')
        meta = cPickle.loads(open(meta_path).read())

        self.n_items_height = np.max(item_ids) + 1
        self.n_items_width = int(meta['n_user'])
        self.n_users_height = np.max(user_ids) + 1
        self.n_users_width = int(meta['n_item'])
        self.n_contents_items = int(meta['n_content_item'])
        self.n_contents_users = int(meta['n_content_user'])

        return (item_row, item_col, item_ids, item_feature, item_contents,
                user_row, user_col, user_ids, user_feature, user_contents,
                item_pref, user_pref)

    def generate(self, train, contents, begin, end):
        for i in range(begin, end):
            column_t    = train[i].indices
            value_t     = train[i].data

            feature_t      = contents[i].indices
            contents_t     = contents[i].data

            if column_t.size == 0 and value_t.size == 0 and feature_t.size == 0 and contents_t.size == 0:
                continue

            yield column_t, value_t, feature_t, contents_t


    def make_db(self, data_dir, out_dir, div):
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        os.makedirs(out_dir)

        data = self.load_data(data_dir, div)
        item_chunk_offsets = self._split_data(opt.chunk_size, self.n_items_height)
        user_chunk_offsets = self._split_data(opt.chunk_size, self.n_users_height)
        item_num_chunks = len(item_chunk_offsets)
        user_num_chunks = len(user_chunk_offsets)
        self.logger.info('Split data into %d chunks for items' % (item_num_chunks))
        self.logger.info('Split data into %d chunks for users' % (user_num_chunks))

        pool = Pool(opt.num_workers)

        try:
            num_data_item = pool.map_async(parse_data, [(cidx, begin, end, data,
                                                        self.n_items_height, self.n_items_width,
                                                        self.n_users_height, self.n_users_width,
                                                        self.n_contents_items,
                                                        self.n_contents_users,
                                                        div, out_dir, 'item')
                                                        for cidx, (begin, end)
                                                        in enumerate(item_chunk_offsets)]).get(999999999)

            num_data_user = pool.map_async(parse_data, [(cidx, begin, end, data,
                                                        self.n_items_height, self.n_items_width,
                                                        self.n_users_height, self.n_users_width,
                                                        self.n_contents_items,
                                                        self.n_contents_users,
                                                        div, out_dir, 'user')
                                                        for cidx, (begin, end)
                                                        in enumerate(user_chunk_offsets)]).get(999999999)

            pool.close()
            pool.join()

        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise

        num_train, num_val = 0, 0
        for train, val in num_data_user:
            num_train += train
            num_val += val

        meta_fout = open(os.path.join(out_dir, 'meta'), 'w')
        meta = {'num_train'     : num_train,
                'num_val'       : num_val,
                'n_item_height' : self.n_items_height,
                'n_user_height' : self.n_users_height}
        meta_fout.write(cPickle.dumps(meta, 2))
        meta_fout.close()

        self.logger.info('size of training set: %s' % num_train)
        self.logger.info('size of validation set: %s' % num_val)
        self.logger.info('n_items: %s' % self.n_items_height)
        self.logger.info('n_users: %s' % self.n_users_height)
        self.logger.info('The number of item contents: %s' % self.n_contents_items)
        self.logger.info('The number of user contents: %s' % self.n_contents_users)

    def _split_data(self, chunk_size, total):
        chunks = [(i, min(i + chunk_size, total))
                  for i in range(0, total, chunk_size)]

        return chunks

    def _byte_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    #TODO: Require Re-Factoring
    def build_data(self, data_dir):
        ''' Remove duplicate interactions and collapse remaining interactions
        into a single binary matrix in the form of a sparse matrix'''

        self.logger.info("Loading a raw data")
        self.user, self.item, self.value = self.load_preference_matrix(
            os.path.join(data_dir, 'train.csv'))

        item_feature = self.load_content_information(
            os.path.join(data_dir, 'item_features_0based.txt'))

        user_feature = self.load_content_information(
            os.path.join(data_dir, 'user_features_0based.txt'))

        self.user_warm_te, self.item_warm_te, self.value_warm_te = self.load_preference_matrix(
            os.path.join(data_dir, 'test_warm.csv'))

        self.user_te_usr_cold, self.item_te_usr_cold, self.value_te_usr_cold = self.load_preference_matrix(
            os.path.join(data_dir, 'test_cold_user.csv'))

        self.user_te_itm_cold, self.item_te_itm_cold, self.value_te_itm_cold = self.load_preference_matrix(
            os.path.join(data_dir, 'test_cold_item.csv'))


        ## Training data
        user_dict, item_dict = self.remove_duplicate()


        self.item_content, self.item_feature, self.item_value, item_content_ids, item_feature_ids, item_content_value = \
            self.remove_contents(item_feature, item_dict)
        self.user_content, self.user_feature, self.user_value, user_content_ids, user_feature_ids, user_content_value = \
            self.remove_contents(user_feature, user_dict)

        self.logger.info('Preference matrix: %d x %d' % (np.max(self.user), np.max(self.item)))
        self.logger.info('Item contents information: %d x %d' % (np.max(self.item_content), np.max(self.item_feature)))
        self.logger.info('User contents information: %d x %d' % (np.max(self.user_content), np.max(self.user_feature)))
        self.save_train_data()

        ## TEST DATA FOR WARM START
        test_warm_item_ids = np.loadtxt(
            os.path.join(data_dir, 'test_warm_item_ids.csv'), dtype='int32')
        test_warm_item_list = [item_dict[i] for i in test_warm_item_ids]

        self.remove_duplicate_test(user_dict, item_dict)
        user_te_tr, item_te_tr, value_te_tr = np.asarray(self.user), np.asarray(self.item), np.asarray(self.value)

        item_based_data, item_test_dict = self.get_train_for_test(self.item_warm_te, test_warm_item_list, item_te_tr, user_te_tr, value_te_tr,
                                                                  self.item_content, self.item_feature, self.item_value, 'item')

        user_based_data, user_test_dict = self.get_train_for_test(self.user_warm_te, test_warm_item_list, user_te_tr, item_te_tr, value_te_tr,
                                                                  self.user_content, self.user_feature, self.user_value, 'user')

        mask_warm = self.get_mask_for_test(item_test_dict, user_test_dict, user_te_tr, item_te_tr, value_te_tr)

        self.save_test_data(item_based_data, user_based_data, mask_warm, self.value_warm_te,
                            './Input/test_warm.h5py')

        ## TEST DATA FOR USER COLD START
        test_cold_user_item_ids = np.loadtxt(
            os.path.join(data_dir, 'test_cold_user_item_ids.csv'), dtype='int32')
        test_cold_user_item_list = [item_dict[i] for i in test_cold_user_item_ids]

        user_dict_cold_user = self.remove_duplicate_cold_user(item_dict)
        user_based_data_cold_user \
            = self.get_contents_for_cold(user_dict_cold_user, user_content_ids, user_feature_ids, user_content_value, self.user_te_usr_cold)

        item_based_data_cold_user, _ = self.get_train_for_test(self.item_te_usr_cold, test_cold_user_item_list, item_te_tr, user_te_tr, value_te_tr,
                                                                                 self.item_content, self.item_feature, self.item_value, 'item')
        mask_cold_user = ([], [], [])

        self.save_test_data(item_based_data_cold_user, user_based_data_cold_user, mask_cold_user, self.value_te_usr_cold,
                            './Input/test_cold_user.h5py')

        ## TEST DATA FOR ITEM COLD START
        test_cold_item_item_ids = np.loadtxt(
            os.path.join(data_dir, 'test_cold_item_item_ids.csv'), dtype='int32')
        # test_cold_item_item_list = [item_dict[i] for i in test_cold_item_item_ids]

        item_dict_cold_item = self.remove_duplicate_cold_item(user_dict, test_cold_item_item_ids)

        item_based_data_cold_item\
            = self.get_contents_for_cold(item_dict_cold_item, item_content_ids, item_feature_ids, item_content_value, self.item_te_itm_cold)


        user_based_data_cold_item, _ = self.get_train_for_test(self.user_te_itm_cold, _, user_te_tr, item_te_tr, value_te_tr,
                                                               self.user_content, self.user_feature, self.user_value, 'user')

        mask_cold_item = ([], [], [])
        self.save_test_data(item_based_data_cold_item, user_based_data_cold_item, mask_cold_item, self.value_te_itm_cold,
                            './Input/test_cold_item.h5py')

        meta_fout = open(os.path.join('./Input', 'meta'), 'w')
        meta = {'n_item'        : np.max(self.item) + 1,
                'n_user'        : np.max(self.user) + 1,
                'n_content_item': np.max(self.item_feature) + 1,
                'n_content_user': np.max(self.user_feature) + 1}
        meta_fout.write(cPickle.dumps(meta, 2))
        meta_fout.close()

        ## Test item index for evaluating

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

    def remove_duplicate(self):
        self.value = np.ones_like(self.value)

        uni = list(set(zip(self.user, self.item, self.value)))
        uni.sort()

        user, item, self.value = zip(*uni)

        user_list = np.unique(user)
        user_dict = dict()
        for i, j in enumerate(user_list):
            user_dict[j] = i

        item_list = np.unique(item)
        item_dict = dict()
        for i, j in enumerate(item_list):
            item_dict[j] = i

        self.user = [user_dict[i] for i in user]
        self.item = [item_dict[i] for i in item]

        return user_dict, item_dict


    def remove_contents(self, data, dictionary):
        data.sort()

        data = np.asarray(data)
        content_ids, feature_ids, value \
            = data[:, 0], data[:, 1], data[:, 2]

        contents = [(i, dictionary[j]) for i, j in enumerate(content_ids) if j in dictionary]
        contents = np.asarray(contents)
        content_ind = contents[:, 0]
        contents    = contents[:, 1]

        features = feature_ids[content_ind]
        values = value[content_ind]

        return contents, features, values, content_ids, feature_ids, value

    def save_train_data(self):
        with h5py.File('./Input/recsys2017_warm.h5py', 'w') as data:
            item_based = data.create_group('item_based')
            # Preference matrix
            pref = item_based.create_group('pref')
            users  = pref.create_dataset('user', np.shape(self.user), 'i')
            items  = pref.create_dataset('item', np.shape(self.item), 'i')
            values = pref.create_dataset('value', np.shape(self.value), 'i')
            users[:], items[:], values[:]  = self.user, self.item, self.value

            # Item Contents
            item_con      = item_based.create_group('contents')
            content_items = item_con.create_dataset('item', np.shape(self.item_content), 'i')
            item_features = item_con.create_dataset('feature', np.shape(self.item_feature), 'i')
            contents      = item_con.create_dataset('value', np.shape(self.item_value), 'f')
            content_items[:], item_features[:], contents[:] \
                = self.item_content, self.item_feature, self.item_value

            user_based = data.create_group('user_based')

            pref = user_based.create_group('pref')
            users  = pref.create_dataset('user', np.shape(self.user), 'i')
            items  = pref.create_dataset('item', np.shape(self.item), 'i')
            values = pref.create_dataset('value', np.shape(self.value), 'i')
            users[:], items[:], values[:]  = self.user, self.item, self.value

            # User Contents
            user_con            = user_based.create_group('contents')
            content_users       = user_con.create_dataset('user', np.shape(self.user_content), 'i')
            user_features       = user_con.create_dataset('feature', np.shape(self.user_feature), 'i')
            contents_user_value = user_con.create_dataset('value', np.shape(self.user_value), 'f')
            content_users[:], user_features[:], contents_user_value[:] \
                = self.user_content, self.user_feature, self.user_value

    def remove_duplicate_test(self, user_dict, item_dict):
        value_warm_te = np.ones_like(self.value_warm_te)

        uni = list(set(zip(self.user_warm_te, self.item_warm_te, value_warm_te)))
        uni.sort()

        self.user_warm_te, self.item_warm_te, self.value_warm_te = zip(*uni)

        self.user_warm_te = [user_dict[i] for i in self.user_warm_te]
        self.item_warm_te = [item_dict[i] for i in self.item_warm_te]

    def get_train_for_test(self, warm_test, item_list, ids, target, value, contents, features, contents_values, type):
        if type == 'user':
            test_list = np.unique(warm_test)
        else:
            test_list = np.unique(item_list)
        test_dict = dict()
        for i, j in enumerate(test_list):
            test_dict[j] = i

        warm_test = [test_dict[i] for i in warm_test]

        ids_tr = [(i, test_dict[j]) for i, j in enumerate(ids) if j in test_dict]
        ids_tr = np.asarray(ids_tr)
        train_ind = ids_tr[:, 0]
        ids_tr    = ids_tr[:, 1]
        target_tr = target[train_ind]
        value_tr  = value[train_ind]

        contents_tr = [(i, test_dict[j]) for i, j in enumerate(contents) if j in test_dict]
        contents_tr = np.asarray(contents_tr)
        train_content_ind = contents_tr[:, 0]
        contents_tr = contents_tr[:, 1]
        features_tr = features[train_content_ind]
        values_tr   = contents_values[train_content_ind]

        return (ids_tr, target_tr, value_tr, contents_tr, features_tr, values_tr, warm_test), test_dict

    def get_mask_for_test(self, item_test_dict, user_test_dict, user, item, value):
        item_ids = [(i, item_test_dict[j]) for i, j in enumerate(item) if j in item_test_dict]
        item_ids = np.asarray(item_ids)
        test_ind = item_ids[:, 0]
        item  = item_ids[:, 1]
        user  = user[test_ind]
        value = value[test_ind]

        user_ids = [(i, user_test_dict[j]) for i, j in enumerate(user) if j in user_test_dict]
        user_ids = np.asarray(user_ids)
        test_ind = user_ids[:, 0]
        user  = user_ids[:, 1]
        item  = item[test_ind]
        value = value[test_ind]

        return (user, item, value)

    def save_test_data(self, item_based_data, user_based_data, mask_warm, target_value, filename):
        item_te_tr_i, user_te_tr_i, value_te_tr_i, contents_te_tr_i, features_te_tr_i, \
            contents_value_te_tr_i, item_warm_te = item_based_data

        user_te_tr_u, item_te_tr_u, value_te_tr_u, contents_te_tr_u, features_te_tr_u, \
            contents_value_te_tr_u, user_warm_te = user_based_data

        mask_warm_user, mask_warm_item, mask_warm_value = mask_warm

        with h5py.File(filename, 'w') as data:
            item_based = data.create_group('item_based')

            self.save_groups(item_based, 'pref',
                             dataset_name = ['user', 'item', 'value'],
                             dataset = [user_te_tr_i, item_te_tr_i, value_te_tr_i],
                             dtypes = ['i', 'i', 'i'])
            # pref = item_based.create_group('pref')
            # users = pref.create_dataset('user', np.shape(user_te_tr_i), 'i')
            # items = pref.create_dataset('item', np.shape(item_te_tr_i), 'i')
            # values = pref.create_dataset('value', np.shape(value_te_tr_i), 'i')
            #
            # users[:], items[:], values[:] = user_te_tr_i, item_te_tr_i, value_te_tr_i

            self.save_groups(item_based, 'contents',
                             dataset_name = ['item', 'feature', 'value'],
                             dataset = [contents_te_tr_i, features_te_tr_i, contents_value_te_tr_i],
                             dtypes = ['i', 'i', 'f'])

            user_based = data.create_group('user_based')

            self.save_groups(user_based, 'pref',
                             dataset_name = ['user', 'item', 'value'],
                             dataset = [user_te_tr_u, item_te_tr_u, value_te_tr_u],
                             dtypes = ['i', 'i', 'i'])

            self.save_groups(user_based, 'contents',
                             dataset_name = ['user', 'feature', 'value'],
                             dataset = [contents_te_tr_u, features_te_tr_u, contents_value_te_tr_u],
                             dtypes = ['i', 'i', 'f'])

            self.save_groups(data, 'mask',
                             dataset_name = ['user', 'item', 'value'],
                             dataset = [mask_warm_user, mask_warm_item, mask_warm_value],
                             dtypes = ['i', 'i', 'i'])

            self.save_groups(data, 'target',
                             dataset_name = ['user', 'item', 'value'],
                             dataset = [user_warm_te, item_warm_te, target_value],
                             dtypes = ['i', 'i', 'i'])

            self.logger.info("Test data for warm start: %d x %d" % (np.max(user_warm_te), np.max(item_warm_te)))

    def remove_duplicate_cold_user(self, item_dict):
        user_te, item_te, value_te = self.user_te_usr_cold, self.item_te_usr_cold, self.value_te_usr_cold
        value_te = np.ones_like(value_te)

        uni = list(set(zip(user_te, item_te, value_te)))
        uni.sort()

        user_te_cold, item_te_cold, self.value_te_usr_cold = zip(*uni)

        user_cold_list = np.unique(user_te_cold)
        user_cold_dict = dict()
        for i, j in enumerate(user_cold_list):
            user_cold_dict[j] = i

        self.user_te_usr_cold = [user_cold_dict[i] for i in user_te_cold]
        self.item_te_usr_cold = [item_dict[i] for i in item_te_cold]

        return user_cold_dict

    def remove_duplicate_cold_item(self, user_dict, item_list):
        user_te, item_te, value_te = self.user_te_itm_cold, self.item_te_itm_cold, self.value_te_itm_cold
        value_te = np.ones_like(value_te)

        uni = list(set(zip(user_te, item_te, value_te)))
        uni.sort()

        user_te, item_te, self.value_te_itm_cold = zip(*uni)

        item_cold_list = np.unique(item_list)
        item_cold_dict = dict()
        for i, j in enumerate(item_cold_list):
            item_cold_dict[j] = i

        self.user_te_itm_cold = [user_dict[i] for i in user_te]
        self.item_te_itm_cold = [item_cold_dict[i] for i in item_te]

        return item_cold_dict

    def get_contents_for_cold(self, cold_dict, content_ids, feature_ids, content_value, temp):
        content_for_cold = [(i, cold_dict[j]) for i, j in enumerate(content_ids) if j in cold_dict]
        content_for_cold = np.asarray(content_for_cold)
        cold_ids = content_for_cold[:, 0]

        user_content_cold = content_for_cold[:, 1]
        user_feature_cold = feature_ids[cold_ids]
        user_value_cold   = content_value[cold_ids]

        return ([], [], [], user_content_cold, user_feature_cold, user_value_cold, temp)

    def save_groups(self, data, group_name, dataset_name, dataset, dtypes):
        group = data.create_group(group_name)

        for i, j, dtype in zip(dataset_name, dataset, dtypes):
            x = group.create_dataset(i, np.shape(j), dtype)
            x[:] = j


if __name__ == '__main__':
    data = Data()
    fire.Fire({'make_db': data.make_db,
               'build_data': data.build_data})