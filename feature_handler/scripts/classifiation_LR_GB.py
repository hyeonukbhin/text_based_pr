#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
from collections import OrderedDict
from collections import namedtuple
from pprintpp import pprint

import gensim
import ast
from sklearn.linear_model import LogisticRegression
import matplotlib.pylab as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

from sklearn.model_selection import KFold, cross_val_score, ShuffleSplit

DATASET_FILEPATH = '../../dataset/mypersonality.csv'
MODEL_FILEPATH = '../../models/'


def read_df(filename):
    df = pd.read_csv(filename, sep=',', na_values=".", index_col=0)
    return df

def get_class(x, trait_name="sEXT"):

    '''
    ,sEXT,sNEU,sAGR,sCON,sOPN
    0.0,1.35,1.25,1.65,1.45,2.25
    0.33,2.9,2.2,3.3,3.2,3.85
    0.67,3.7,2.85,3.94,3.9,4.45
    1.0,5.0,4.75,5.0,5.0,5.0
    '''

    filename = "../../model_interfacer/scripts/3c_percentile.csv"
    boundary = read_df(filename)
    # print(boundary)

    thres_0 = boundary[trait_name][0.]
    thres_33 = boundary[trait_name][0.33]
    thres_67 = boundary[trait_name][0.67]
    thres_1 = boundary[trait_name][1.]

    x = float(x)

    if x < thres_33:
        label = 0
    elif x < thres_67:
        label = 1
    else:
        label = 2

    # result = [thres_0, thres_33, thres_66, thres_1]

    return label

# def get_class_list(x_list, trait_list=["sEXT","sNEU","sAGR","sCON","sOPN"]):
#
#     '''
#     ,sEXT,sNEU,sAGR,sCON,sOPN
#     0.0,1.35,1.25,1.65,1.45,2.25
#     0.33,2.9,2.2,3.3,3.2,3.85
#     0.67,3.7,2.85,3.94,3.9,4.45
#     1.0,5.0,4.75,5.0,5.0,5.0
#     '''
#
#     filename = "../../model_interfacer/scripts/3c_percentile.csv"
#     boundary = read_df(filename)
#     # print(boundary)
#
#     # thres_0 = boundary[trait_name][0.]
#     # thres_33 = boundary[trait_name][0.33]
#     # thres_67 = boundary[trait_name][0.67]
#     # thres_1 = boundary[trait_name][1.]
#
#     thres_0 = [boundary[trait][0.] for trait in trait_list]
#     thres_33 = [boundary[trait][0.33] for trait in trait_list]
#     thres_67 = [boundary[trait][0.67] for trait in trait_list]
#     thres_1 = [boundary[trait][1.] for trait in trait_list]
#
#     # x = float(x)
#
#
#
#     for i in range(len(x_list)):
#         ifx[i] <
#
#     if x < thres_33:
#         label = 0
#     elif x < thres_67:
#         label = 1
#     else:
#         label = 2
#
#     # result = [thres_0, thres_33, thres_66, thres_1]
#
#     return label

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

def map_list_type(l, dtype=str):
    return list(map(dtype, l))

def main():
    # 모델 가져오기
    model = gensim.models.Doc2Vec.load(MODEL_FILEPATH+"personality_doc2vec.model")

    # 데이터 가져오기
    df_train_docs = read_df("train_docs.csv")
    train_sentence_docs = df_train_docs["tokens"].tolist()
    train_rating_docs = df_train_docs["p_score"].tolist()

    train_docs = [(ast.literal_eval(sentence), ast.literal_eval(train_rating_docs[i])) for i, sentence in
                  enumerate(train_sentence_docs)]

    # (float(ast.literal_eval(p_scores)[0]))
    # for p_scores in cv_rating_docs


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
    # print(cv_ext_docs)
    cv_ext_docs = list(OrderedDict.fromkeys(cv_ext_docs).keys())
    # print(cv_ext_docs)


    i= 0
    for idx, doctag in sorted(model.docvecs.doctags.items(), key=lambda x: x[1].offset):
        # print("sEXT : {}, DocTag : {}, offset : {}".format(cv_ext_docs[i], idx, doctag))
        i += 1


    cv_neu_docs = [(float(ast.literal_eval(p_scores)[1])) for p_scores in cv_rating_docs]
    cv_agr_docs = [(float(ast.literal_eval(p_scores)[2])) for p_scores in cv_rating_docs]
    cv_con_docs = [(float(ast.literal_eval(p_scores)[3])) for p_scores in cv_rating_docs]
    cv_opn_docs = [(float(ast.literal_eval(p_scores)[4])) for p_scores in cv_rating_docs]
    # print(cv_ext_docs)

    cv_ext_docs = np.asarray(cv_ext_docs, dtype=np.float32)
    cv_lext_docs = np.log(cv_ext_docs + 1)


    TaggedDocument = namedtuple('TaggedDocument', 'words tags')
    tagged_train_docs = [TaggedDocument(d, map_list_type(c, dtype=float)) for d, c in train_docs]
    tagged_test_docs = [TaggedDocument(d, map_list_type(c, dtype=float)) for d, c in test_docs]


    # # pprint(tagged_train_docs)
    # for doc in tagged_train_docs:
    #     print("tttt")
    #     print(doc.words)
    #
    #     print(type(doc.words))

    # ,sEXT,sNEU,sAGR,sCON,sOPN


    train_x = [model.infer_vector(doc.words) for doc in tagged_train_docs]
    # train_y_3c = [get_class(doc.tags[0], trait_name="sEXT") for doc in tagged_train_docs]
    train_y_3c_ext = [get_class(doc.tags[0], trait_name="sEXT") for doc in tagged_train_docs]
    train_y_3c_neu = [get_class(doc.tags[1], trait_name="sNEU") for doc in tagged_train_docs]
    train_y_3c_agr = [get_class(doc.tags[2], trait_name="sAGR") for doc in tagged_train_docs]
    train_y_3c_con = [get_class(doc.tags[3], trait_name="sCON") for doc in tagged_train_docs]
    train_y_3c_opn = [get_class(doc.tags[4], trait_name="sOPN") for doc in tagged_train_docs]
    # train_y = [float(doc.tags[0]) for doc in tagged_train_docs]


    # pprint(train_x)
    # pprint(train_y_3c)
    test_x = [model.infer_vector(doc.words) for doc in tagged_test_docs]
    test_y_3c_ext = [get_class(doc.tags[0], trait_name="sEXT") for doc in tagged_test_docs]
    test_y_3c_neu = [get_class(doc.tags[1], trait_name="sNEU") for doc in tagged_test_docs]
    test_y_3c_agr = [get_class(doc.tags[2], trait_name="sAGR") for doc in tagged_test_docs]
    test_y_3c_con = [get_class(doc.tags[3], trait_name="sCON") for doc in tagged_test_docs]
    test_y_3c_opn = [get_class(doc.tags[4], trait_name="sOPN") for doc in tagged_test_docs]
    # test_y = [float(doc.tags[0]) for doc in tagged_test_docs]



    print("====== logistic regression ======")
    # for
    classifier_lr_ext = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    classifier_lr_neu = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    classifier_lr_agr = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    classifier_lr_con = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    classifier_lr_opn = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    # classifier = LogisticRegression(C=1e5)
    # logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

    classifier_lr_ext.fit(train_x, train_y_3c_ext)
    classifier_lr_neu.fit(train_x, train_y_3c_neu)
    classifier_lr_agr.fit(train_x, train_y_3c_agr)
    classifier_lr_con.fit(train_x, train_y_3c_con)
    classifier_lr_opn.fit(train_x, train_y_3c_opn)


    result_lr_ext = classifier_lr_ext.score(test_x, test_y_3c_ext)
    result_lr_neu = classifier_lr_neu.score(test_x, test_y_3c_neu)
    result_lr_agr = classifier_lr_agr.score(test_x, test_y_3c_agr)
    result_lr_con = classifier_lr_con.score(test_x, test_y_3c_con)
    result_lr_opn = classifier_lr_opn.score(test_x, test_y_3c_opn)

    print(result_lr_ext)
    print(result_lr_neu)
    print(result_lr_agr)
    print(result_lr_con)
    print(result_lr_opn)

    joblib.dump(classifier_lr_ext, MODEL_FILEPATH+"lr_ext.sav")
    joblib.dump(classifier_lr_neu, MODEL_FILEPATH+"lr_neu.sav")
    joblib.dump(classifier_lr_agr, MODEL_FILEPATH+"lr_agr.sav")
    joblib.dump(classifier_lr_con, MODEL_FILEPATH+"lr_con.sav")
    joblib.dump(classifier_lr_opn, MODEL_FILEPATH+"lr_opn.sav")

    loaded_model = joblib.load(MODEL_FILEPATH+"lr_ext.sav")

    reresult_lr_ext = loaded_model.score(test_x, test_y_3c_ext)

    print(reresult_lr_ext)

    print(test_x[0])
    label = loaded_model.predict([test_x[0]])
    print(label)
    # lr_result = loaded_model.predict(test_x[0])
    # print(lr_result)


    print("====== Gradient Boosting Classfication regression ======")
    # classifier_gb = GradientBoostingClassifier(n_estimators=100, random_state=0, learning_rate=0.3)
    classifier_gb_ext = GradientBoostingClassifier(n_estimators=100)
    classifier_gb_neu = GradientBoostingClassifier(n_estimators=100)
    classifier_gb_agr = GradientBoostingClassifier(n_estimators=100)
    classifier_gb_con = GradientBoostingClassifier(n_estimators=100)
    classifier_gb_opn = GradientBoostingClassifier(n_estimators=100)


    classifier_gb_ext.fit(train_x, train_y_3c_ext)
    classifier_gb_neu.fit(train_x, train_y_3c_neu)
    classifier_gb_agr.fit(train_x, train_y_3c_agr)
    classifier_gb_con.fit(train_x, train_y_3c_con)
    classifier_gb_opn.fit(train_x, train_y_3c_opn)


    # result_gb = classifier_gb.score(test_x, test_y_3c)
    result_gb_ext = classifier_gb_ext.score(test_x, test_y_3c_ext)
    result_gb_neu = classifier_gb_neu.score(test_x, test_y_3c_neu)
    result_gb_agr = classifier_gb_agr.score(test_x, test_y_3c_agr)
    result_gb_con = classifier_gb_con.score(test_x, test_y_3c_con)
    result_gb_opn = classifier_gb_opn.score(test_x, test_y_3c_opn)

    print(result_gb_ext)
    print(result_gb_neu)
    print(result_gb_agr)
    print(result_gb_con)
    print(result_gb_opn)

    joblib.dump(classifier_gb_ext, MODEL_FILEPATH+"gb_ext.sav")
    joblib.dump(classifier_gb_neu, MODEL_FILEPATH+"gb_neu.sav")
    joblib.dump(classifier_gb_agr, MODEL_FILEPATH+"gb_agr.sav")
    joblib.dump(classifier_gb_con, MODEL_FILEPATH+"gb_con.sav")
    joblib.dump(classifier_gb_opn, MODEL_FILEPATH+"gb_opn.sav")


    loaded_model = joblib.load(MODEL_FILEPATH+"gb_ext.sav")

    reresult_gb_ext = loaded_model.score(test_x, test_y_3c_ext)

    print(reresult_gb_ext)



main()