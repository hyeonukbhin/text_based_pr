# -*- coding: utf-8 -*-

import pandas as pd
from collections import OrderedDict
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
DM = 1 # PVDM if use PVDBOW 사용하려면 0

ext_score_buff = 0

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

def map_list_type(l, dtype=str):
    return list(map(dtype, l))



# def tokenize(df, authid, doc_index):
def tokenize(df, authid):

    global ext_score_buff, doc_index
    df_id = df[df['#AUTHID'] == authid]
    status_list = df_id["STATUS"].tolist()
    cleaned_tokens_2d_list = [nltk.word_tokenize(cleanText(sentence)) for sentence in status_list]
    splited_token_list = [t for d in cleaned_tokens_2d_list for t in d]
    p_score_list = map_list_type(df_id[['sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN']].iloc[0].values.tolist(), dtype=str)
    # p_score_list = map_list_type(df_id[['sEXT']].iloc[0].values.tolist(), dtype=str)
    # print(token_list)
    import spacy
    # spacy.require_gpu()
    nlp = spacy.load('en')
    splited_token_list = [term for term in splited_token_list if term not in nlp.Defaults.stop_words]

    ext_score = df_id['sEXT'].iloc[0]
    if ext_score_buff == ext_score:
        doc_index = doc_index
    else:
        doc_index += 1
    ext_score_buff = ext_score

    neu_score = df_id['sNEU'].iloc[0]
    agr_score = df_id['sAGR'].iloc[0]
    con_score = df_id['sCON'].iloc[0]
    opn_score = df_id['sOPN'].iloc[0]

    # p_score_list = [ext_score,neu_score,agr_score,con_score,opn_score]
    doc_name = "Post_{}".format(doc_index - 1)

    # print("p_score list : {}".format(p_score_list))
    return splited_token_list, p_score_list, doc_name


def save_df(df, filename):
    df.to_csv(filename, mode="w", sep=',')


def main():
    global doc_index
    train_df = read_df(DATASET_FILEPATH)

    df_comp = train_df[["#AUTHID", 'STATUS', 'sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN', 'NETWORKSIZE']]
    df_comp = df_comp.sort_values(["sEXT"], ascending=[True])


    df_ext_low_sample = df_comp[:50]
    df_ext_high_sample = df_comp[-50:]

    df_ext_sample = df_ext_low_sample.append(df_ext_high_sample)
    # data = data.append(data_row, ignore_index=True)
    save_df(df_ext_sample, "df_ext_sample.csv")
    # print

    print(df_ext_low_sample)
    # print(df_comp)
    # pprint(df_comp[:100])

    # scoure_list = list(set(df_comp["sEXT"]))
    # print("e score list :")
    # pprint(scoure_list)
    # pprint(len(scoure_list))
    # print("length of df_comp : {}".format(len(authid_list)))


    authid_list = list(OrderedDict.fromkeys(df_comp["#AUTHID"]).keys())

    print(df_comp["#AUTHID"])
    print(authid_list)

    print("length of df_comp(non-drop) : {}".format(len(authid_list)))
    df_comp = drop_under_N_sentence(df_comp, authid_list, THRES_ID)
    authid_list = list(OrderedDict.fromkeys(df_comp["#AUTHID"]).keys())

    print("length of df_comp(droped) : {}".format(len(authid_list)))

    # authid_list = list(set(df_comp["#AUTHID"]))

    train_authid_list = authid_list[:BOUND_TT]
    test_authid_list = authid_list[BOUND_TT:]

    # train_docs = [(tokenize(df_comp, authid, index)) for index, authid in enumerate(train_authid_list)]
    # test_docs = [(tokenize(df_comp, authid, index)) for index, authid in enumerate(test_authid_list)]
    # both_docs = [(tokenize(df_comp, authid, index)) for index, authid in enumerate(authid_list)]

    doc_index = 0
    train_docs = [(tokenize(df_comp, authid)) for index, authid in enumerate(train_authid_list)]
    doc_index = 0
    test_docs = [(tokenize(df_comp, authid)) for index, authid in enumerate(test_authid_list)]
    doc_index = 0
    both_docs = [(tokenize(df_comp, authid)) for index, authid in enumerate(authid_list)]


    # print(train_docs[0][0][0])
    # pprint(both_docs[:2])
    print(len(both_docs))

    df_train = pd.DataFrame(train_docs)
    df_train.columns = ["tokens", "p_score", "Post_index"]

    df_test = pd.DataFrame(test_docs)
    df_test.columns = ["tokens", "p_score", "Post_index"]

    df_both = pd.DataFrame(both_docs)
    df_both.columns = ["tokens", "p_score", "Post_index"]


    save_df(df_train, "train_docs.csv")
    save_df(df_test, "test_docs.csv")
    save_df(df_both, "both_docs.csv")


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

    tagged_train_docs = [TaggedDocument(tokens, [post_idx]) for tokens, p_scores, post_idx in train_docs]
    tagged_test_docs = [TaggedDocument(tokens, [post_idx]) for tokens, p_scores, post_idx in test_docs]
    tagged_both_docs = [TaggedDocument(tokens, [post_idx]) for tokens, p_scores, post_idx in both_docs]


    print(tagged_train_docs[0])

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
                                  dm=DM,
                                  compute_loss=True)

    model.build_vocab(tagged_both_docs)
    model.train(tagged_both_docs, epochs=model.epochs, total_examples=model.corpus_count)

    model.save("personality_doc2vec.model")
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    print(model.most_similar(positive=["apple"], topn=5))

    # print(model.most_similar(positive=['pediatrician', 'tooth'], negative=['baby'], topn=5))

    print(model.docvecs.most_similar('Post_10', topn=5))

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