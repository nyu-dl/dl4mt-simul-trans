Learning to Translate in Real-time with Neural Machine Translation
===================================
Translation in Real-time, a.k.a, Simultaneous Translation.

This code is the Theano implementation of the EACL2017 paper [Learning to Translate in Real-time with Neural Machine Translation](https://arxiv.org/abs/1610.00388). It is based on the dl4mt-tutorial (https://github.com/nyu-dl/dl4mt-tutorial).

Dependencies:
----------------------
### Python 2.7
* Theano 0.8.2 (cuda 8.0, cudnn v5)
* seaborn, pandas (for drawing the heatmap)
* NLTK 3.2.1

### Preprocessing
The preprocessing and evaluation scripts are from [MOSES](https://github.com/moses-smt/mosesdecoder).

Dataset:
----------------------
We used the WMT'15 corpora as our training set for both pretraining the NMT model and training the Simultaneous NMT model.
The original WMT'15 corpora can be downloaded from [here](http://www.statmt.org/wmt15/translation-task.html). 
For the preprocessed corpora used in our experiments, we used the same dataset of dl4mt-c2c (https://github.com/nyu-dl/dl4mt-c2c). Both the source and target datasets are preprocessed using byte-pair encoding (http://arxiv.org/abs/1508.07909, https://github.com/rsennrich/subword-nmt).
* [Standard version (for bilingual models, 3.5GB)](https://drive.google.com/open?id=0BxmEQ91VZAPQam5pc2ltQ1BBTTQ)

Pretraining:
----------------------
TBA.

* We provide the pretrained RU-EN, CS-EN and DE-EN unidirectional models for both directions as follows:
TBA.
