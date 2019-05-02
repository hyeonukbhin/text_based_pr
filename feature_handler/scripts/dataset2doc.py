# -*- coding: utf-8 -*-

import pandas as pd

from collections import namedtuple
from pprintpp import pprint
import ast
import re
import nltk

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

THRES_ID = 3
BOUND_TT = 180
DATASET_FILEPATH = '../../dataset/mypersonality.csv'


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


train_df = read_df(DATASET_FILEPATH)
df_comp = train_df[["#AUTHID", 'STATUS', 'sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN', 'NETWORKSIZE']]

authid_list = list(set(df_comp["#AUTHID"]))
print("length of df_comp {}".format(len(authid_list)))
df_comp = drop_under_N_sentence(df_comp, authid_list, THRES_ID)
authid_list = list(set(df_comp["#AUTHID"]))
print("length of df_comp === {}".format(len(authid_list)))

train_authid_list = authid_list[:BOUND_TT]
test_authid_list = authid_list[BOUND_TT:]


# print(len(train_authid_list))
# print(len(test_authid_list))

# tokenized_text = [nltk.word_tokenize(cleanText(sentence)) for sentence in test]
# test_result = [t for d in tokenized_text for t in d]
#
# print(tokenized_text)
# print(test_result)
# print("asdfasdf")
# test_result = [t for d in status_list for t in d]

# print(test_result)

def tokenize(df, authid):
    df_id = df_comp[df_comp['#AUTHID'] == authid]
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


train_docs = [(tokenize(df_comp, authid)) for authid in train_authid_list]
test_docs = [(tokenize(df_comp, authid)) for authid in test_authid_list]


def save_df(df, filename):
    df.to_csv(filename, mode="w", sep=',')


def main():
    df_train = pd.DataFrame(train_docs)
    df_train.columns = ["tokens", "p_score"]

    df_test = pd.DataFrame(test_docs)
    df_test.columns = ["tokens", "p_score"]

    save_df(df_train, "train_docs.csv")
    save_df(df_test, "test_docs.csv")

    # pprint(train_docs)

    print("ttttt")

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

    # pprint(tagged_train_docs[:20])

    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from nltk.tokenize import word_tokenize

    # data = ["I love machine learning. Its awesome.",
    #         "I love coding in python",
    #         "I love building chatbots",
    #         "they chat amagingly well"]
    #
    # tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

    # pprint(tagged_data)

    import multiprocessing

    cores = multiprocessing.cpu_count()

    # doc2vec parameters

    print(cores)
    vector_size = 300

    window_size = 15

    word_min_count = 2

    sampling_threshold = 1e-5

    negative_size = 5

    train_epoch = 10

    dm = 0  # {0:pvdbow, 1:pvdm}

    worker_count = cores  # number of parallel processes

    import os

    import gensim
    # from gensim.models import doc2vec
    # 사전 구축

    model = gensim.models.Doc2Vec(vector_size=300,
                                  window=5,
                                  seed=1234,
                                  negative=20,
                                  min_count=5,
                                  workers=worker_count,
                                  alpha=0.025,
                                  min_alpha=0.025,
                                  epochs=20,
                                  # dm=1,
                                  compute_loss=True)

    model.build_vocab(tagged_train_docs)
    model.train(tagged_train_docs, epochs=model.epochs, total_examples=model.corpus_count)
    # print(model.get_latest_training_loss())
    # print(model.compute_loss)
    # print(model.comment)
    # model.train()

    # model2 = gensim.models.Word2Vec(size=300,
    #                               window=5,
    #                               seed=1234,
    #                               negative=20,
    #                               min_count=5,
    #                               workers=worker_count,
    #                               alpha=0.025,
    #                               min_alpha=0.025,
    #                               iter=20,
    #                               compute_loss=True)

    # print("\nEvaluating %s" % model)
    # err_rate, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
    # error_rates[str(model)] = err_rate
    # print("\n%f %s\n" % (err_rate, model))
    # print(model.get_latest_training_loss())
    # tsne

    # model.get
    # doc_vectorizer = doc2vec.Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, seed=1234, total_examples=doc_vectorizer.corpus_count)
    # doc_vectorizer.build_vocab(tagged_train_docs)
    # # Train document vectors!
    # for epoch in range(10):
    #     doc_vectorizer.train(tagged_train_docs)
    #     doc_vectorizer.alpha -= 0.002  # decrease the learning rate
    #     doc_vectorizer.min_alpha = doc_vectorizer.alpha  # fix the learning rate, no decay
    # To save

    model.save("personality_doc2vec.model")

    print("end======")

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