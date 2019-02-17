import numpy as np
import h5py

from scipy.stats import rankdata
from scipy.sparse import csr_matrix

def get_eval(preds, mode, meta):

    if mode == 'warm':
        data = h5py.File('./Input/test_warm.h5py', 'r')
    elif mode == 'cold_user':
        data = h5py.File('./Input/test_cold_user.h5py', 'r')
    else:
        data = h5py.File('./Input/test_cold_item.h5py', 'r')

    mask_user, mask_item, mask_value = \
        data['mask']['user'][:], data['mask']['item'][:], data['mask']['value'][:]

    target_user, target_item, target_value = \
        data['target']['user'][:], data['target']['item'][:], data['target']['value'][:]

    mask   = csr_matrix((mask_value, (mask_user, mask_item)),
                        shape=(meta['n_user_height'], meta['n_item_height']))
    target = csr_matrix((target_value, (target_user, target_item)),
                        shape=(meta['n_user_height'], meta['n_item_height']))

    print('\n')
    max_user = np.min((10000, meta['n_user_height']))
    target=target[0:max_user]
    preds=preds[0:max_user, :]
    mask=mask[0:max_user]
    recalls = get_recall(target, preds, mask, np.arange(50, 550, 50))

    for k, recall in zip(np.arange(50, 550, 50), recalls):
        print("[*] RECALL@%d: %.4f" % (k, recall))

#TODO: Multicore processing Using Pool
def get_recall(target, preds, mask, n_recalls):
    # ratingTest[:, [1, 0]] = ratingTest[:, [0, 1]]


    # temp = np.zeros((16980, 5551))
    # temp[(ratingTest[:, 0], ratingTest[:, 1])] = 1
    # preds       = np.transpose(preds)
    # target      = np.transpose(ratingTest)
    # mask        = np.transpose(mask)
    preds  = np.asarray(preds)
    # target = target.toarray()
    mask   = mask.toarray()
    print(np.sort(preds[0, :])[::-1][:100])
    print(np.sort(preds[0, :] * target[0].toarray()[0])[::-1])

    preds       = preds * (1-mask) - 100 * mask
    non_zero_idx = np.asarray(target.sum(axis=1)).flatten() != 0
    #
    del mask
    preds   = preds[non_zero_idx, :]
    target  = target[non_zero_idx]

    # pred_user_interest = pred_user_interest * test_mask + (1 - test_mask) * (-100)
    preds = get_order_array(preds)

    recall = []
    for i in n_recalls:
        pred_user_interest = preds <= i

        match_interest  = target.multiply(pred_user_interest)
        num_match       = np.sum(match_interest, axis=1, dtype=np.float32)
        num_interest    = target.sum(axis=1)

        user_recall = num_match / num_interest
        recall.append(np.average(user_recall))

    return recall

def get_order_array(list):
    order = np.empty(list.shape, dtype=int)
    for k, row in enumerate(list):
        order[k] = rankdata(-row, method='ordinal') - 1

    return order