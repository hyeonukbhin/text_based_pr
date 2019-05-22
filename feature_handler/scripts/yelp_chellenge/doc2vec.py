# -*- coding: utf-8 -*-

import os
import codecs
import json
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
sns.set_context('notebook', font_scale=1.5)

# %matplotlib inline
import matplotlib.pyplot as plt


data_directory = os.path.join('/home/kist/yelp_13')
intermediate_directory = os.path.join(data_directory, 'intermediate')

df = pd.read_csv(os.path.join(intermediate_directory, 'useful.csv'))
luseful = np.log(df['useful'] + 1)

# print(df[:10])

import thinc_gpu_ops


# Gensim's Doc2Vec class creates vector representations for entire documents
# The input object is an iterator of LineSentence objects
# The default dm=1 refers to the distributed memory algorithm
# The algorithm runs through sentences twice: (1) build the vocab,
# (2) learn a vector representation for each word and for each label (sentence)
# Better results can be achieved by iterating over the data several times

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

doc2vec_filepath = os.path.join(intermediate_directory, 'doc2vec_model')


doc2vec = Doc2Vec.load(doc2vec_filepath)

print(doc2vec.most_similar(positive=["pediatrician"], topn=5))

print(doc2vec.most_similar(positive=['pediatrician', 'tooth'], negative=['baby'], topn=5))

print(doc2vec.docvecs.most_similar('Review_10', topn=5))


