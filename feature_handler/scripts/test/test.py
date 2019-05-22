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




