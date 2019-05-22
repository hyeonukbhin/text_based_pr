# -*- coding: utf-8 -*-
from pprintpp import pprint
import gensim
import pandas as pd
from sklearn.manifold import TSNE

from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim
import gensim.models as g
import numpy as np
import ast
from collections import namedtuple

def read_df(filename):
    df = pd.read_csv(filename, sep=',', na_values=".", index_col=0)
    return df






def main():


    model = gensim.models.Doc2Vec.load("personality_doc2vec.model")
    print("tttt")
    # print(model["1.25"])
    # print(model.docvecs[0])
    # print(model.docvecs.count)

    # pprint(model.docvecs.doctags["2.55"])
    for idx, doctag in sorted(model.docvecs.doctags.items(), key=lambda x: x[1].offset):
        print(idx, doctag)

    print(model.most_similar(positive=["king"], topn=5))

    print(model.most_similar(positive=['king', 'woman'], negative=['man'], topn=5))

    # pprint(model.docvecs.offset2doctag)
    # pprint(model.docvecs.doctag_syn0)
    # pprint("aaaa")
    # pprint(model.docvecs['4.55'])
    # pprint(model.docvecs.most_similar('4.55'))
    # pprint(model.wv['hello'])

    # pprint(model.docvecs.max_rawint)
    # [-0.8885772 - 0.24909 - 1.1056005   0.8084259   0.28025487  0.22296177

    # pprint(model.docvecs.index2doctag)
    # pprint(model.docvecs.index_to_doctag(0))
    # model.docvecs.
    # print(model)


    # pprint(model.docvecs.doctags)
    # print(model.docvecs.int_index())
    # print(model.docvecs["4.6"][:10])
    # print(model.docvecs[0][:10])


    # TaggedDocument(words=
    #                ['A', 'textbook', 'a', 'week', 'argh', 'Still', 'recovering', 'I', 'look', 'like', 'a', 'big', 'fly0_0', 'can', 'read', 'with', 'his', 'right', 'eye', 'now', 'No', 'worries', 'Happy', 'Thanksgiving', 'everyone', 'Addicted', 'to', 'Les', 'Miserables', 'Block', '2', 'is', 'over', 'Time', 'to', 'relax', 'Finally', 'a', 'three', 'day', 'weekend', 'to', 'catch', 'up', 'with', 'school', 'Moving', 'is', 'such', 'a', 'hassle', 'Memory', 'All', 'alone', 'in', 'the', 'moonlight', 'I', 'can', 'smile', 'at', 'the', 'old', 'days', 'When', 'the', 'dawn', 'comes', 'Tonight', 'will', 'be', 'a', 'memory', 'too', 'And', 'a', 'new', 'day', 'will', 'begin'], tags=[2.5, 3.5, 3.75, 3.0, 4.5]),

    # print(model.docvecs[1].doctags)


    # 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
    mpl.rcParams['axes.unicode_minus'] = False

    df_train_docs = read_df("train_docs.csv")
    train_sentence_docs = df_train_docs["tokens"].tolist()
    train_rating_docs = df_train_docs["p_score"].tolist()
    # print(train_rating_docs)
    train_docs = [(ast.literal_eval(sentence), ast.literal_eval(train_rating_docs[i])) for i, sentence in
                  enumerate(train_sentence_docs)]

    df_test_docs = read_df("test_docs.csv")
    test_sentence_docs = df_test_docs["tokens"].tolist()
    test_rating_docs = df_test_docs["p_score"].tolist()
    test_docs = [(ast.literal_eval(sentence), ast.literal_eval(test_rating_docs[i])) for i, sentence in
                 enumerate(test_sentence_docs)]


    TaggedDocument = namedtuple('TaggedDocument', 'words tags')
    tagged_train_docs = [TaggedDocument(d, c) for d, c in train_docs]
    tagged_test_docs = [TaggedDocument(d, c) for d, c in test_docs]


    # pprint(model.docvecs)
    # print
    # train_x = [model.infer_vector(doc.words) for doc in tagged_train_docs]
    # for doc in tagged_train_docs:
    #     print("tttt")
        # print(doc.words)
        # print(model.docvecs(doc.words))


    # # pprint(tagged_train_docs)
    # train_x = [model.infer_vector(doc.words) for doc in tagged_train_docs]
    #
    # print(train_x[0])
    # X_train = np.array([model.docvecs[train_x]])

    # print(X_train[:10])


    # tagged_train_docs = [TaggedDocument(d, c) for d, c in train_docs]
    #
    #
    # X_train = np.array([model.docvecs[str(i)] for i in range(len(tagged_tr))])
    # test_x = [model.infer_vector(doc.words) for doc in tagged_test_docs]
    # ['fie', 'Time', 'Inc', 'has', 'blocked', 'gchat', 'does', 'this', 'mean', 'I', 'have', 'to', 'be', 'productive',
    #  'now', 'is', 'in', 'search', 'of', 'the', 'perfect', 'margarita', 'I', 'was', 'sitting', 'at', 'LaGuardia',
    #  'wondering', 'why', 'my', 'flight', 'was', 'delayed', 'and', 'then', 'I', 'saw', 'on', 'the', 'TV', 'that',
    #  'there', 'had', 'been', 'a', 'bomb', 'threat', 'and', 'evacuation', 'earlier', 'today', 'lucky', 'meeee', 'do',
    #  'i', 'have', 'to', 'take', 'out', 'my', 'nose', 'ring', 'and', 'join', 'the', 'corporate', 'world', 'just',
    #  'spent', 'the', 'last', 'hour', 'looking', 'at', 'photos', 'from', 'junior', 'abroad', 'in', 'london', 'and', 'is',
    #  'dying', 'to', 'go', 'back', 'hmmm', 'i', 'better', 'not', 'fail', 'my', 'passfail', 'is', 'struggling', 'ewww',
    #  '33', 'of', 'snow', 'apparently', 'the', 'most', 'at', 'one', 'time', 'since', '1969', 'LAW', 'SCHOOL', 'IS',
    #  'TOO', 'HARD', 'thanks', 'everyone', 'but', 'i', 'decided', 'to', 'get', 'younger', 'this', 'year']

    # vector = model.wv.get_vector("hello my name is hyeonuk")
    # y_train = train['label']




main()
