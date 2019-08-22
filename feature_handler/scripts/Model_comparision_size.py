#-*- coding: utf-8 -*-

import pandas as pd
from collections import OrderedDict

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

    print(cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'), -1)
    print(type(cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')))
    return np.mean(cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))*-1
    # return np.sqrt(np.mean(np.multiply(cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error'), -1)))


# neg_mean_absolute_error = neg_mean_absolute_error_scorer,
# neg_mean_squared_error = neg_mean_squared_error_scorer,
# neg_mean_squared_log_error = neg_mean_squared_log_error_scorer,

def cv_accuracy(name, model, X, y, cv=5):
    """ Compute an overall RMSE across all folds of cross validation"""

    # print(name)
    result = np.mean(cross_val_score(model, X, y, cv=cv))
    # if name is "gbr_d2v" or "gbr_d2v" or "gbr_d2v":
    #
    # if (name == "gradient boosting reg.") or (name == "random forest reg.") or (name == "Xboosting reg."):
    #     result = result*-1

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

    x_label = [0,10,20,30,40,50]
    y_label = ["Logistic Regression", "Random Forest", "Support Vector Machine", "Gradiant Boosting"]
    data = [[0,0.43,0.534,0.54,0.58,0.43],
            [0,0.51,0.634,0.62,0.65,0.72],
            [0,0.51,0.704,0.71,0.72,0.74],
            [0,0.45,0.724,0.73,0.74,0.76]]
    # table = [[]]
    # for model, in y_label:
    #     print(model)
    # for model, in y_labe
    # print(type(y_label))
    # for y in y_label:
    #     print(y)
    df_test = pd.DataFrame(data, columns=x_label, index=y_label, dtype=float)

    # print(df_test)
    # print(df_test[x_label[0]][y_label[0]])
    # table = df_test[x_label[0]][y_label[0]]
    table = []

    # for y in y_label:
    #     print(model)
    for model_name in y_label:
        for train_size in x_label:
            table.append({'model': model_name,
                          'F1-score': df_test[train_size][model_name],
                          'train_size': train_size})
    df = pd.DataFrame(table)
    print(df)

    # font = {'family': 'normal',
    #         'size': 10}

    # import matplotlib
    # matplotlib.rc('font', **font)
    plt.rcParams.update({'font.size': 15})

    # import matplotlib.font_manager as fm
    # TimesNewRoman = fm.FontProperties(fname='/Library/Fonts/Times New Roman.ttf')

    # print(plt.rcParams.items())
    # plt.rcParams["font.family"] = TimesNewRoman
    # print(df_test)
    plt.figure(figsize=(12, 5))
    fig = sns.pointplot(x='train_size', y='F1-score', hue='model', data=df)
    sns.set_context('notebook', font_scale=2.0)
    fig.set(ylabel='F1-score')
    # plt.ylabel('F1-score', fontproperties=TimesNewRoman)

    fig.set(xlabel='The number of sentences')
    fig.set(title='F1-score of the big-5 traits according to the number of sentences')
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