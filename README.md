

# BAE_HCF

Tensorflow implementation for reproducing the Hybrid recoomender system results on CiteULike and RecSys datset in the paper <a href="http://kalman.kaist.ac.kr/assets/papers/MLSP-2.pdf">"Basis Learning Autoencoders for Hybrid Collaborative Filtering in Cold Start Setting"</a>. If you want to reproduce the results on Recsys, change the branch with "RecSys".



#### Requirements

Python2.7, tensorflow-gpu >= 1.14.0, Numpy, Scipy



## Preparation tf.records format

This code uses a pipeline based on the tf.records format. Therefore, it is necessary to convert given raw datasets to the tf.records format. After the preparation, 5-fold datasets which are made of different seeds are created.

For *citeulike-a* dataset

- Warm start

```python ./utils/preprocessing.py make_db Input/citeulike/citeulike-a.h5py Input/citeulike-a/warm 'warm'```

- Cold start

```python ./utils/prerocessing.py make_db Input/citeulike/citeulike-a.h5py Input/citeulike-a/cold 'cold'```

For *citeulike-t* dataset

- Warm start

```python ./utils/preprocessing.py make_db Input/citeulike/citeulike-t.h5py Input/citeulike-t/warm 'warm'```

- Cold start

```python ./utils/preprocessing.py make_db Input/citeulike/citeulike-t.h5py Input/citeulike-t/cold 'cold'```