# -*- coding: utf-8 -*-

import pandas as pd

from collections import namedtuple
from pprintpp import pprint
import ast
import re
import nltk
import sklearn
import multiprocessing
import gensim

# nltk.download('punkt')

'''
#AUTHID
STATUS : 1 facebook post

sEXT, sNEU, sAGR, sCON, sOPN : 0 ~ 5 BFi-44 Score
cEXT, cNEU, cAGR, cCON, cOPN : 0 or 1 Binary Class normalized(뭐 기준인지 확인행)
DATE
NETWORKSIZE
BETWEENNESS
NBETWEENNESS
DENSITY
BROKERAGE
NBROKERAGE
TRANSITIVITY
'''

''' STATIC Parameter Define '''

THRES_ID = 5
BOUND_TT = 180
DATASET_FILEPATH = '../../dataset/mypersonality.csv'

VECTOR_SIZE = 300
WINDOW_SIZE = 5
SEED = 1234
NEGATIVE_SIZE = 20
WORD_MIN_COUNT = 5
LEARNING_ALPHA = 0.025
LEARNING_NEG_ALPHA = 0.025
EPOCHES = 20

def read_df(filename):
    # df = pd.read_csv(filename, sep=',', na_values=".",index_col=0, encoding = "ISO-8859-1")
    df = pd.read_csv(filename, sep=',', na_values=".", encoding="ISO-8859-1")
    return df


def drop_under_N_sentence(df, authid_list, drop_n):
    for authid in authid_list:
        df_id = df[df['#AUTHID'] == authid]
        count = len(df_id.index)
        if count < drop_n:
            df = df[df['#AUTHID'] != authid]
    return df


def cleanText(readData):
    # 텍스트에 포함되어 있는 특수 문자 제거
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
    return text


def tokenize(df, authid):
    df_id = df[df['#AUTHID'] == authid]
    status_list = df_id["STATUS"].tolist()
    cleaned_tokens_2d_list = [nltk.word_tokenize(cleanText(sentence)) for sentence in status_list]
    splited_token_list = [t for d in cleaned_tokens_2d_list for t in d]
    p_score_list = df_id[['sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN']].iloc[0].values.tolist()
    # print(token_list)
    ext_score = df_id['sEXT'].iloc[0]
    neu_score = df_id['sNEU'].iloc[0]
    agr_score = df_id['sAGR'].iloc[0]
    con_score = df_id['sCON'].iloc[0]
    opn_score = df_id['sOPN'].iloc[0]
    # p_score_list = [ext_score,neu_score,agr_score,con_score,opn_score]

    return splited_token_list, p_score_list


def save_df(df, filename):
    df.to_csv(filename, mode="w", sep=',')


def main():
    train_df = read_df(DATASET_FILEPATH)
    df_comp = train_df[["#AUTHID", 'STATUS', 'sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN', 'NETWORKSIZE']]

    authid_list = list(set(df_comp["#AUTHID"]))
    print("length of df_comp : {}".format(len(authid_list)))
    df_comp = drop_under_N_sentence(df_comp, authid_list, THRES_ID)
    authid_list = list(set(df_comp["#AUTHID"]))
    print("length of df_comp : {}".format(len(authid_list)))

    train_authid_list = authid_list[:BOUND_TT]
    test_authid_list = authid_list[BOUND_TT:]

    train_docs = [(tokenize(df_comp, authid)) for authid in train_authid_list]
    test_docs = [(tokenize(df_comp, authid)) for authid in test_authid_list]

    df_train = pd.DataFrame(train_docs)
    df_train.columns = ["tokens", "p_score"]

    df_test = pd.DataFrame(test_docs)
    df_test.columns = ["tokens", "p_score"]

    save_df(df_train, "train_docs.csv")
    save_df(df_test, "test_docs.csv")


    TaggedDocument = namedtuple('TaggedDocument', 'words tags')

    # # 여기서는 15만개 training documents 전부 사용함

    def act_fuction(x):
        if x > 0.5:
            y = 1
        else:
            y = 0
        return y

    # tagged_train_docs = [TaggedDocument(d, [str(act_fuction(c[0]))]) for d, c in train_docs]
    # tagged_test_docs = [TaggedDocument(d, [str(act_fuction(c[0]))]) for d, c in test_docs]

    tagged_train_docs = [TaggedDocument(d, c) for d, c in train_docs]
    tagged_test_docs = [TaggedDocument(d, c) for d, c in test_docs]


    worker_count = multiprocessing.cpu_count()

    print("CPU Cores : {}".format(worker_count))

    model = gensim.models.Doc2Vec(vector_size=VECTOR_SIZE,
                                  window=WINDOW_SIZE,
                                  seed=SEED,
                                  negative=NEGATIVE_SIZE,
                                  min_count=WORD_MIN_COUNT,
                                  workers=worker_count,
                                  alpha=LEARNING_ALPHA,
                                  min_alpha=LEARNING_NEG_ALPHA,
                                  epochs=EPOCHES,
                                  # dm=1,
                                  compute_loss=True)

    model.build_vocab(tagged_train_docs)
    model.train(tagged_train_docs, epochs=model.epochs, total_examples=model.corpus_count)

    model.save("personality_doc2vec.model")

    print("======end======")

    # document처리할것
    #
    # document1 id에 따라서
    # document2 personality score에 따라서
    #
    #
    # 같은 성격 점수를 갖고 있는 사람들끼리
    #
    # 같은 성격 클래스 갖고 있는 사람들 끼리 클래스 처리 할것

main()