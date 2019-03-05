import numpy as np

from scipy.stats import rankdata

def get_recall(ratingTest, preds, test_mask, n_recalls):
    # ratingTest[:, [1, 0]] = ratingTest[:, [0, 1]]


    # temp = np.zeros((16980, 5551))
    # temp[(ratingTest[:, 0], ratingTest[:, 1])] = 1
    preds       = np.transpose(preds)
    test_mask   = np.transpose(test_mask)
    temp        = np.transpose(ratingTest)
    # temp      = np.asarray(ratingTest)
    # preds     = np.asarray(preds)
    # test_mask = np.asarray(test_mask)

    non_zero_idx = np.sum(temp, axis=1) != 0

    pred_user_interest   = preds[non_zero_idx, :]
    target_user_interest = temp[non_zero_idx, :]
    test_mask            = test_mask[non_zero_idx, :]

    pred_user_interest = pred_user_interest * test_mask + (1 - test_mask) * (-100)
    pred_user_interest = get_order_array(pred_user_interest)
    pred_user_interest = pred_user_interest <= n_recalls

    match_interest  = pred_user_interest * target_user_interest
    num_match       = np.sum(match_interest, axis=1)
    num_interest    = np.sum(target_user_interest, axis=1)

    user_recall = num_match / num_interest
    recall = np.average(user_recall)
    return recall

def get_order_array(list):
    order = np.empty(list.shape, dtype=int)
    for k, row in enumerate(list):
        order[k] = rankdata(-row, method='ordinal') - 1

    return order