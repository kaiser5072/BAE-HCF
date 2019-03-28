import numpy as np
import fire

def average(idx):
    f = open('results/results%s.txt' % idx, 'r')

    result = []
    for line in f:
        result.append(line.split())

    result = np.asarray(result).astype(np.float32)
    results = np.average(result, axis=0)

    print('\n')
    for k, recall in enumerate(results):
        print("[*] RECALL@%d: %.4f" % (k, recall))

if __name__ == '__main__':
    fire.Fire({'average': average})