#-*- coding: utf-8 -*-

import pandas as pd

from collections import namedtuple
from pprintpp import pprint

import gensim

model = gensim.models.Doc2Vec.load("personality_doc2vec.model")


similar_words = ["sex", "apple", "man"]
for similar_word in similar_words:
    print("similar : {}".format(similar_word))
    pprint(model.most_similar(similar_word, topn=10))


# doc2vec parameter iteration 돌려봐

# 파일 정리하고


# model.sim
# print("similar : me")
#
# pprint(model.most_similar('me'))
#
print("similar : football + hand - foot")

pprint(model.most_similar(positive=['football', 'hand'], negative=['foot']))

print("similar : king + woman - man")

pprint(model.most_similar(positive=['woman', 'king'], negative=['man']))
# => [('악당/Noun', 0.32974398136138916),
#     ('곽지민/Noun', 0.305545836687088),
#     ('심영/Noun', 0.2899821400642395),
#     ('오빠/Noun', 0.2856029272079468),
#     ('전작/Noun', 0.2840743064880371),
#     ('눈썹/Noun', 0.28247544169425964),
#     ('광팬/Noun', 0.2795347571372986),
#     ('지능/Noun', 0.2794691324234009),
#     ('박보영/Noun', 0.27567577362060547),
#     ('강예원/Noun', 0.2734225392341614)]

# pprint(model.most_similar('왕/Noun'))


def read_df(filename):
    df = pd.read_csv(filename, sep=',', na_values=".", index_col=0)
    return df

import ast

df_train_docs = read_df("train_docs.csv")
train_sentence_docs = df_train_docs["tokens"].to_list()
train_rating_docs = df_train_docs["p_score"].to_list()

train_docs = [(ast.literal_eval(sentence), ast.literal_eval(train_rating_docs[i])) for i, sentence in enumerate(train_sentence_docs)]

df_test_docs = read_df("test_docs.csv")
test_sentence_docs = df_test_docs["tokens"].to_list()
test_rating_docs = df_test_docs["p_score"].to_list()
test_docs = [(ast.literal_eval(sentence), ast.literal_eval(test_rating_docs[i])) for i, sentence in enumerate(test_sentence_docs)]


TaggedDocument = namedtuple('TaggedDocument', 'words tags')
# 여기서는 15만개 training documents 전부 사용함
tagged_train_docs = [TaggedDocument(d, c) for d, c in train_docs]
tagged_test_docs = [TaggedDocument(d, c) for d, c in test_docs]


def act_fuction(x):
    if x > 3:
        y = 1
    else:
        y = 0
    return y


# pprint(tagged_train_docs)
train_x = [model.infer_vector(doc.words) for doc in tagged_train_docs]
train_y_binary = [act_fuction(doc.tags[0]) for doc in tagged_train_docs]
train_y = [doc.tags[0] for doc in tagged_train_docs]

test_x = [model.infer_vector(doc.words) for doc in tagged_test_docs]
test_y_binary = [act_fuction(doc.tags[0]) for doc in tagged_test_docs]
test_y = [doc.tags[0] for doc in tagged_test_docs]



from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
encoded_train_y = lab_enc.fit_transform(train_y)
encoded_test_y = lab_enc.fit_transform(test_y)
# >>> array([1, 3, 2, 0], dtype=int64)

# pprint(train_x)
# pprint(train_y)

# for yy in train_y:
#     print(type(yy))
#     print(yy)




from sklearn.linear_model import LogisticRegression
print("logistic regression======")
classifier = LogisticRegression(random_state=1234)
classifier.fit(train_x, train_y_binary)
result = classifier.score(test_x, test_y_binary)
# classifier.
print(result)
# => 0.78246000000000004

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

def plot_sklearn(model):
    plt.scatter(train_x, train_y)
    xx = np.linspace(0, 1, 1000)
    plt.plot(xx, model.predict(xx[:, np.newaxis]))
    plt.show()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


# linear regression
print("linear regression======")
# poly = PolynomialFeatures(9)
regression_model = LinearRegression()
# regression_model = make_pipeline(poly, LinearRegression())
regression_model.fit(train_x, train_y)
# print(regression_model.steps[1][1].coef_)
# plot_sklearn(regression_model)
# summary = regression_model.summary()
# print(summary)
result = regression_model.score(test_x, test_y)
result2 = regression_model.score(train_x, train_y)
print(result)


# reg.score(X, y)
# classifier.
# print(result2)

# ridge regression
print("ridge regression======")
regression_model = Ridge(alpha=1, random_state=1234)
regression_model.fit(train_x, train_y)
result = regression_model.score(test_x, test_y)
# result2 = regression_model.score(train_x, train_y)
print(result)
# regression_model = make_pipeline(poly, Ridge(alpha=0.01)).fit(train_x, train_y)
# print(regression_model.steps[1][1].coef_)
# plot_sklearn(regression_model)
#
# # lasso regression
#
# model = make_pipeline(poly, Lasso(alpha=0.01)).fit(X, y)
# print(model.steps[1][1].coef_)
# plot_sklearn(model)
#
# model = make_pipeline(poly, ElasticNet(alpha=0.01, l1_ratio=0.5)).fit(X, y)
# print(model.steps[1][1].coef_)
# plot_sklearn(model)
#
# # elasticNet regression
#


from sklearn.pipeline import Pipeline



# d2v_models = [("lr_d2v", lr_d2v), ("ridge_d2v", ridge_d2v)]

from sklearn.model_selection import KFold, cross_val_score, ShuffleSplit

def cv_rmse(model, X, y, cv=5, scoring='neg_mean_squared_error'):
    """ Compute an overall RMSE across all folds of cross validation"""

    return np.sqrt(np.mean(np.multiply(cross_val_score(
        model, X, y, cv=cv, scoring='neg_mean_squared_error'), -1)))


def RMSE(y_true, y_pred):
    """ Root Mean Squared Error"""

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def RMSLE(y_true, y_pred):
    """ Root Mean Squared Logarithmic Error"""

    return np.sqrt(np.mean(((np.log(y_true + 1) - np.log(y_pred + 1)) ** 2)))



# test_x = test_x[:20]
# test_y = test_y[:20]


alphas = [0.01, 0.05, 0.1, 0.5, 1, 5, 500]
for alpha in alphas:
    lr_d2v = Pipeline([("lr", LinearRegression())])
    ridge_d2v = Pipeline([("ridge", Ridge(alpha=alpha))])

    lr_rmse = RMSE(np.array(test_y), lr_d2v.fit(train_x,train_y).predict(test_x))
    ridge_rmse = RMSE(np.array(test_y), ridge_d2v.fit(train_x,train_y).predict(test_x))

    print("lr_rmse")
    print(lr_rmse)
    print(lr_d2v.score(test_x,test_y))
    print("ridge_rmse alpha = {}".format(alpha))
    print(ridge_rmse)
    print(ridge_d2v.score(test_x,test_y))


# d2v_rmse = [(name, cv_rmse(d2v_model, np.array(model.docvecs), train_y, cv=5)) for name, d2v_model in d2v_models]
# from tabulate import tabulate
#
#
#
# print(tabulate(sorted(d2v_rmse, key=lambda x:x[1]), floatfmt=".4f", headers=("model", "RMSE_5cv")))