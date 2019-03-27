import numpy as np
import fire

def average(idx):
    f = open('results/results%d.txt' % idx, 'r')

    result = []
    for line in f:
        result.append(line.split())

    result = np.asarray(result).astype(np.float32)
    print(result)
    results = np.average(result, axis=0)

    for k, recall in enumerate(results):
        print("[*] RECALL@%d: %.4f" % (k, recall))

if __name__ == '__main__':
    fire.Fire({'average': average})