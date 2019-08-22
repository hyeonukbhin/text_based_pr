# Author: Robert Guthrie
# Translator: Don Kim

test1 = ["1", "2", '3', "1", "2"]
test2 = [4, 1, 2, 3, 1, 2]

import collections

authid_list1 = list(collections.OrderedDict.fromkeys(test2).keys())

authid_list2 = list(set(test2))





print(authid_list1)
print(authid_list2)
# print(test2.)

#
# def map_list_type(list, type):
#     result = map(type, list)
#
#     return result
#
# def map_list_type(l, dtype=str):
#     return list(map(dtype, l))
#
# print(map_list_type(test2, str))
# list(map(int,test1)) # => [1,2,3]
# # print(
#     list(map(str,test2)) # => [1,2,3]

# import gensim

import codecs

import os
data_directory = "/home/kist/Desktop"

review_json_filepath = os.path.join(data_directory, 'review.json')
doc2vec_filepath = "doc2vec_model_test"
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, ShuffleSplit


# a = np.array([-3.77649239, -0.97781948, -0.52422557, -1.33490285, -3.98226499], dtype="float64")

a = np.array([-1.5, -.5, -0.5, -0.5, -0.5], dtype="float64")

result = np.sqrt(np.mean(np.multiply(a, -1)))

print(result)
# X = []
# with codecs.open(trigram_reviews_filepath, encoding='utf-8') as f:
#     for review in f:
#         X.append(review)
#
# labels = ['Review_' + str(x) for x in range(len(X))]
#
#

from gensim.models import Doc2Vec

from gensim.models.doc2vec import LabeledSentence




