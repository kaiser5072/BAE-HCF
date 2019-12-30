## BAE_HCF

Tensorflow implementation for reproducing the Hybrid recoomender system results on CiteULike and RecSys dataset in the paper ["Basis Learning Autoencoders for Hybrid Collaborative Filtering in Cold Start Setting"](http://kalman.kaist.ac.kr/assets/papers/MLSP-2.pdf). If you want to reproduce the results on **CiteULike**, change the branch with <a href="https://github.com/kaiser5072/BAE-HCF/tree/citeulike">"citeulike"</a>.

</br>

**Requirements**

Python2.7, tensorflow-gpu == 1.14.0, Numpy, Scipy

</br>

### Dataset

We use the datasets from recsys 2017 challenge.

- Preference matrix consists of **1,064,237 users** and **495,601 jobs**
- User contents information contains **830x1** vector for each user
- Item contents information contains **2,737x1** vector for each item
- Test data for warm start contains 124,960 users and 62,434 jobs
- Test data for user cold start
- Test data for item cold start

</br>

### Preparation tfrecords format

We use tensorflow input pipeline using tfrecords format. Therefore, we convert given raw datasets to the tfrecords format.

First, we clean ukp given raw data, and save it with .h5py format.

```python utils/preprocessing.py vectorize 'train'```



- Training set

  ```python data.py make_db './Input/recsys2017_warm_h5py './Input/tfrecords/train```

- Test set for warm start

  ```python data.py make_db './Input/test_warm.h5py' './Input/tfrecords/warm'```

- Test set for user cold start

  ```python data.py make_db './Input/test_cold_user.h5py' './Input/tfrecords/cold_user'```

- Test set for item cold start

  ```python data.py make_db './Input/test_cold_item.h5py' './Input/tfrecords/cold_item'```

</br>

### Training



### Test

- Warm start
- User cold start
- Item cold start