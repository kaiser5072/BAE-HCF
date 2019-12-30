## BAE_HCF

Tensorflow implementation for reproducing the Hybrid recoomender system results on CiteULike and RecSys dataset in the paper ["Basis Learning Autoencoders for Hybrid Collaborative Filtering in Cold Start Setting"](http://kalman.kaist.ac.kr/assets/papers/MLSP-2.pdf). If you want to reproduce the results on **CiteULike**, change the branch with <a href="https://github.com/kaiser5072/BAE-HCF/tree/citeulike">"citeulike"</a>.



**Requirements**

Python2.7, tensorflow-gpu == 1.14.0, Numpy, Scipy



### Dataset

We use the datasets from recsys 2017 challenge.



### Preparation tfrecords format

```python utils/preprocessing.py vectorize 'train'```

