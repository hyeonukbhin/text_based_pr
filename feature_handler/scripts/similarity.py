# -*- coding: utf-8 -*-
from pprintpp import pprint
import gensim

model = gensim.models.Doc2Vec.load("personality_doc2vec.model")

similar_words = ["sex", "apple", "man"]

for similar_word in similar_words:
    print("similar : {}".format(similar_word))
    pprint(model.most_similar(similar_word, topn=10))


print("similar : football + hand - foot")

pprint(model.most_similar(positive=['football', 'hand'], negative=['foot']))

print("similar : king + woman - man")

pprint(model.most_similar(positive=['woman', 'king'], negative=['man']))
