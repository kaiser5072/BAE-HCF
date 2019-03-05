import numpy as np

from scipy.stats import rankdata

def get_recall(target, preds, mask, n_recalls):
    # ratingTest[:, [1, 0]] = ratingTest[:, [0, 1]]


    # temp = np.zeros((16980, 5551))
    # temp[(ratingTest[:, 0], ratingTest[:, 1])] = 1
    preds  = np.transpose(preds)
    mask   = np.transpose(mask)
    target = np.transpose(target)
    # temp      = np.asarray(ratingTest)
    # preds     = np.asarray(preds)
    # test_mask = np.asarray(test_mask)

    preds        = preds * (1-mask) - 100 * mask
    non_zero_idx = np.sum(target, axis=1) != 0

    preds  = preds[non_zero_idx, :]
    target = target[non_zero_idx, :]

    preds = get_order_array(preds)

    recall = []
    for i in n_recalls:
        pred_user_interest = preds <= i

        match_interest  = pred_user_interest * target
        num_match       = np.sum(match_interest, axis=1, dtype=np.float32)
        num_interest    = np.sum(target, axis=1)

        user_recall = num_match / num_interest
        recall.append(np.average(user_recall))

    for k, recall in zip(np.arange(50, 550, 50), n_recalls):
        print("[*] RECALL@%d: %.4f" % (k, recall))

    return recall

def get_order_array(list):
    order = np.empty(list.shape, dtype=int)
    for k, row in enumerate(list):
        order[k] = rankdata(-row, method='ordinal') - 1

    return order