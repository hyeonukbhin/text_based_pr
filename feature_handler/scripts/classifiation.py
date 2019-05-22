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

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, ShuffleSplit
from sklearn.metrics import mean_squared_error, make_scorer
from collections import defaultdict
from tabulate import tabulate
import seaborn as sns



def read_df(filename):
    df = pd.read_csv(filename, sep=',', na_values=".", index_col=0)
    return df

def act_fuction(x):
    if x > 3:
        y = 1
    else:
        y = 0
    return y



def plot_sklearn(train_x, train_y, model):
    plt.scatter(train_x, train_y)
    xx = np.linspace(0, 1, 1000)
    plt.plot(xx, model.predict(xx[:, np.newaxis]))
    plt.show()


def cv_rmse(model, X, y, cv=5, scoring='neg_mean_squared_error'):
    """ Compute an overall RMSE across all folds of cross validation"""

    return np.sqrt(np.mean(np.multiply(cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'), -1)))

def cv_accuracy(name, model, X, y, cv=5):
    """ Compute an overall RMSE across all folds of cross validation"""

    # print(name)
    result = np.mean(cross_val_score(model, X, y, cv=cv))
    # if name is "gbr_d2v" or "gbr_d2v" or "gbr_d2v":
    return result

def RMSE(y_true, y_pred):
    """ Root Mean Squared Error"""

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def RMSLE(y_true, y_pred):
    """ Root Mean Squared Logarithmic Error"""

    return np.sqrt(np.mean(((np.log(y_true + 1) - np.log(y_pred + 1)) ** 2)))

def benchmark(model, X, y, n):
    ss = ShuffleSplit(n_splits=5, test_size=1 - n, random_state=0)
    scores = []
    for train, test in ss.split(X, y):
        scores.append(RMSE(np.array(y[test]), model.fit(X[train], y[train]).predict(X[test])))

        # print("best score :{}".format(model.))
        # print("test sixze :{}".format(len(test)))
    return np.mean(scores)

def main():
    model = gensim.models.Doc2Vec.load("personality_doc2vec.model")




    df_train_docs = read_df("train_docs.csv")
    train_sentence_docs = df_train_docs["tokens"].tolist()
    train_rating_docs = df_train_docs["p_score"].tolist()

    train_docs = [(ast.literal_eval(sentence), ast.literal_eval(train_rating_docs[i])) for i, sentence in
                  enumerate(train_sentence_docs)]

    # (float(ast.literal_eval(p_scores)[0]))
    # for p_scores in cv_rating_docs
    # i= 0
    # for idx, doctag in sorted(model.docvecs.doctags.items(), key=lambda x: x[1].offset):
    #     i += 1
    #     print("sEXT : {}, DocTag : {}, offset : {}".format(train_rating_docs[i][2:5], idx, doctag))

    # pprint(train_docs[0])

    df_test_docs = read_df("test_docs.csv")
    test_sentence_docs = df_test_docs["tokens"].tolist()
    test_rating_docs = df_test_docs["p_score"].tolist()
    test_docs = [(ast.literal_eval(sentence), ast.literal_eval(test_rating_docs[i])) for i, sentence in
                 enumerate(test_sentence_docs)]


    df_cv_docs = read_df("both_docs.csv")
    cv_rating_docs = df_cv_docs["p_score"].tolist()
    # print(cv_rating_docs)

    # 'sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN'
    cv_ext_docs = [(float(ast.literal_eval(p_scores)[0])) for p_scores in cv_rating_docs]
    cv_neu_docs = [(float(ast.literal_eval(p_scores)[1])) for p_scores in cv_rating_docs]
    cv_agr_docs = [(float(ast.literal_eval(p_scores)[2])) for p_scores in cv_rating_docs]
    cv_con_docs = [(float(ast.literal_eval(p_scores)[3])) for p_scores in cv_rating_docs]
    cv_opn_docs = [(float(ast.literal_eval(p_scores)[4])) for p_scores in cv_rating_docs]
    # print(cv_ext_docs)

    cv_ext_docs = np.asarray(cv_ext_docs, dtype=np.float32)
    cv_lext_docs = np.log(cv_ext_docs + 1)

    # print(cv_lext_docs)

    # print(df_both_docs)

    # test_sentence_docs = df_test_docs["tokens"].tolist()
    # test_rating_docs = df_test_docs["p_score"].tolist()
    # test_docs = [(ast.literal_eval(sentence), ast.literal_eval(test_rating_docs[i])) for i, sentence in
    #              enumerate(test_sentence_docs)]



    TaggedDocument = namedtuple('TaggedDocument', 'words tags')
    tagged_train_docs = [TaggedDocument(d, c) for d, c in train_docs]
    tagged_test_docs = [TaggedDocument(d, c) for d, c in test_docs]


    # pprint(tagged_train_docs)
    # for doc in tagged_train_docs:
    #     print("tttt")
    #     print(doc.words)

    # train_y_binary = [act_fuction(doc.tags[0]) for doc in tagged_train_docs]
    # train_y = [doc.tags[0] for doc in tagged_train_docs]
    #
    # test_x = [model.infer_vector(doc.words) for doc in tagged_test_docs]
    # test_y_binary = [act_fuction(doc.tags[0]) for doc in tagged_test_docs]
    # test_y = [doc.tags[0] for doc in tagged_test_docs]


    train_x = [model.infer_vector(doc.words) for doc in tagged_train_docs]
    # train_y_binary = [act_fuction(doc.tags) for doc in tagged_train_docs]
    train_y = [float(doc.tags[0]) for doc in tagged_train_docs]

    test_x = [model.infer_vector(doc.words) for doc in tagged_test_docs]
    # test_y_binary = [act_fuction(doc.tags) for doc in tagged_test_docs]
    test_y = [float(doc.tags[0]) for doc in tagged_test_docs]

    lr_d2v = Pipeline([("lr", LinearRegression())])
    ridge_d2v = Pipeline([("ridge", Ridge(alpha=1))])
    gbr_d2v = Pipeline([("gbr", GradientBoostingRegressor(n_estimators=100))])
    rfr_d2v = Pipeline([("rfr", RandomForestRegressor(n_estimators=100))])
    xgb_d2v = Pipeline([("xgb", XGBRegressor(n_estimators=100))])

    # d2v_models = [("lr_d2v", lr_d2v), ("ridge_d2v", ridge_d2v), ("gbr_d2v", gbr_d2v), ("rfr_d2v", rfr_d2v),
    #               ("xgb_d2v", xgb_d2v)]
    d2v_models = [("linear reg.", lr_d2v), ("ridge reg.", ridge_d2v), ("gradient boosting reg.", gbr_d2v), ("random forest reg.", rfr_d2v),
                  ("Xboosting reg.", xgb_d2v)]
    # d2v_models = [("lr_d2v", lr_d2v), ("ridge_d2v", ridge_d2v)]

    # print(type(doc2vec.docvecs[0]))
    # print(doc2vec.docvecs.doctag_syn0)
    # print(type(np.array(doc2vec.docvecs)))


    d2v_rmse = [(name, cv_rmse(reg_model, model.docvecs.vectors_docs, cv_ext_docs, cv=5))
                for name, reg_model in d2v_models]

    # print(tabulate(sorted(d2v_rmse, key=lambda x: x[1]), floatfmt=".4f", headers=("model", "RMSE_5cv")))
    print(tabulate(d2v_rmse, floatfmt=".4f", headers=("model", "RMSE_5cv")))


    d2v_acc = [(name, cv_accuracy(name, reg_model, model.docvecs.vectors_docs, cv_ext_docs, cv=5))
                for name, reg_model in d2v_models]

    print(tabulate(d2v_acc, floatfmt=".4f", headers=("model", "Accuracy_5cv")))



    train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
    table = []

    for name, reg_model in d2v_models:
        for n in train_sizes:
            table.append({'model': name,
                          'RMSE': benchmark(reg_model, model.docvecs.vectors_docs, cv_ext_docs, n),
                          'train_size': n})
    df = pd.DataFrame(table)

    plt.figure(figsize=(12, 5))
    fig = sns.pointplot(x='train_size', y='RMSE', hue='model',
                        data=df)
    sns.set_context('notebook', font_scale=1.5)
    fig.set(ylabel='RMSE')
    fig.set(xlabel='Training Size')
    fig.set(title='Model Comparison By Training Size')
    plt.show()

    #
    # print(len(test_x))
    # print(len(test_y))
    # print(test_y)
    # # lab_enc = preprocessing.LabelEncoder()
    # # encoded_train_y = lab_enc.fit_transform(train_y)
    # # encoded_test_y = lab_enc.fit_transform(test_y)
    #
    # print("====== logistic regression ======")
    # # classifier = LogisticRegression(random_state=1234)
    # # classifier.fit(train_x, train_y_binary)
    # # result = classifier.score(test_x, test_y_binary)
    # # print(result)
    #
    # # linear regression
    # print("======linear regression======")
    # regression_model = LinearRegression()
    # regression_model.fit(train_x, train_y)
    # result = regression_model.score(test_x, test_y)
    # result2 = regression_model.score(train_x, train_y)
    # print(result)
    #
    # # ridge regression
    # print("======ridge regression======")
    # regression_model = Ridge(alpha=1, random_state=1234)
    # regression_model.fit(train_x, train_y)
    # result = regression_model.score(test_x, test_y)
    # print(result)
    # # test_x = test_x[:20]
    # # test_y = test_y[:20]
    #
    # alphas = [0.01, 0.05, 0.1, 0.5, 1, 5, 500]
    # for alpha in alphas:
    #     lr_d2v = Pipeline([("lr", LinearRegression())])
    #     ridge_d2v = Pipeline([("ridge", Ridge(alpha=alpha))])
    #
    #     lr_rmse = RMSE(np.array(test_y), lr_d2v.fit(train_x, train_y).predict(test_x))
    #     ridge_rmse = RMSE(np.array(test_y), ridge_d2v.fit(train_x, train_y).predict(test_x))
    #
    #     print("lr_rmse")
    #     print(lr_rmse)
    #     print(lr_d2v.score(test_x, test_y))
    #     print("ridge_rmse alpha = {}".format(alpha))
    #     print(ridge_rmse)
    #     print(ridge_d2v.score(test_x, test_y))
    #
    # print(model.docvecs.vector_size)
    #
    # print("aaaaaaaaaaaaa")

main()