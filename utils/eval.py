import numpy as np

from scipy.stats import rankdata

def get_recall(target, preds, mask, n_recalls):
    preds  = np.transpose(preds)
    mask   = np.transpose(mask)
    target = np.transpose(target)

    print(np.sort(preds[0, :])[::-1][:100])

    preds = preds * mask - 100 * (1-mask)
    non_zero_idx = np.sum(target, axis=1) != 0

    preds   = preds[non_zero_idx, :]
    target_user_interest = target[non_zero_idx, :]

    preds = get_order_array(preds)

    recalls = []
    for i in n_recalls:
        pred_user_interest = preds <= i

        match_interest  = pred_user_interest * target_user_interest
        num_match       = np.sum(match_interest, axis=1)
        num_interest    = np.sum(target_user_interest, axis=1)

        user_recall = num_match / num_interest
        recalls.append(np.average(user_recall))

    for k, recall in zip(n_recalls, recalls):
        print("[*] RECALL@%d: %.4f" % (k, recall))

    f = open('results.txt', 'a')
    for recall in recalls:
        f.write('%.4f ' % recall)
    f.write('\n')
    f.close()

    return recall

def get_order_array(list):
    order = np.empty(list.shape, dtype=int)
    for k, row in enumerate(list):
        order[k] = rankdata(-row, method='ordinal') - 1

    return order