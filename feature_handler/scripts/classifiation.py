#-*- coding: utf-8 -*-

import pandas as pd

from collections import namedtuple
from pprintpp import pprint

import gensim
import ast
from sklearn import preprocessing
from sklearn import utils
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, ShuffleSplit


def read_df(filename):
    df = pd.read_csv(filename, sep=',', na_values=".", index_col=0)
    return df

def act_fuction(x):
    if x > 3:
        y = 1
    else:
        y = 0
    return y



def plot_sklearn(model):
    plt.scatter(train_x, train_y)
    xx = np.linspace(0, 1, 1000)
    plt.plot(xx, model.predict(xx[:, np.newaxis]))
    plt.show()




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




def main():
    model = gensim.models.Doc2Vec.load("personality_doc2vec.model")

    df_train_docs = read_df("train_docs.csv")
    train_sentence_docs = df_train_docs["tokens"].tolist()
    train_rating_docs = df_train_docs["p_score"].tolist()

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

    # pprint(tagged_train_docs)
    train_x = [model.infer_vector(doc.words) for doc in tagged_train_docs]
    train_y_binary = [act_fuction(doc.tags[0]) for doc in tagged_train_docs]
    train_y = [doc.tags[0] for doc in tagged_train_docs]

    test_x = [model.infer_vector(doc.words) for doc in tagged_test_docs]
    test_y_binary = [act_fuction(doc.tags[0]) for doc in tagged_test_docs]
    test_y = [doc.tags[0] for doc in tagged_test_docs]

    lab_enc = preprocessing.LabelEncoder()
    encoded_train_y = lab_enc.fit_transform(train_y)
    encoded_test_y = lab_enc.fit_transform(test_y)

    print("====== logistic regression ======")
    classifier = LogisticRegression(random_state=1234)
    classifier.fit(train_x, train_y_binary)
    result = classifier.score(test_x, test_y_binary)
    print(result)

    # linear regression
    print("======linear regression======")
    regression_model = LinearRegression()
    regression_model.fit(train_x, train_y)
    result = regression_model.score(test_x, test_y)
    result2 = regression_model.score(train_x, train_y)
    print(result)

    # ridge regression
    print("======ridge regression======")
    regression_model = Ridge(alpha=1, random_state=1234)
    regression_model.fit(train_x, train_y)
    result = regression_model.score(test_x, test_y)
    print(result)
    # test_x = test_x[:20]
    # test_y = test_y[:20]

    alphas = [0.01, 0.05, 0.1, 0.5, 1, 5, 500]
    for alpha in alphas:
        lr_d2v = Pipeline([("lr", LinearRegression())])
        ridge_d2v = Pipeline([("ridge", Ridge(alpha=alpha))])

        lr_rmse = RMSE(np.array(test_y), lr_d2v.fit(train_x, train_y).predict(test_x))
        ridge_rmse = RMSE(np.array(test_y), ridge_d2v.fit(train_x, train_y).predict(test_x))

        print("lr_rmse")
        print(lr_rmse)
        print(lr_d2v.score(test_x, test_y))
        print("ridge_rmse alpha = {}".format(alpha))
        print(ridge_rmse)
        print(ridge_d2v.score(test_x, test_y))

    print(model.docvecs.vector_size)

    print("aaaaaaaaaaaaa")

main()