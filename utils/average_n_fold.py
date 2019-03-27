import numpy as np

f = open('results.txt', 'r')

result = []
for line in f:
    result.append(line.split())

result = np.asarray(result).astype(np.float32)
print(result)
results = np.average(result, axis=1)

for k, recall in enumerate(results):
    print("[*] RECALL@%d: %.4f" % (k, recall))