


import pandas as pd
import nltk
import ast
from pprintpp import pprint

def read_df(filename):
    df = pd.read_csv(filename, sep=',', na_values=".", index_col=0)
    return df

df_train_docs = read_df("train_docs.csv")
df_test_docs = read_df("test_docs.csv")

train_docs = df_train_docs["sentence"].to_list()
train_docs = [ast.literal_eval(t) for t in train_docs]

test_docs = df_test_docs["sentence"].to_list()
test_docs = [ast.literal_eval(t) for t in test_docs]


train_docs = train_docs[:10]

pprint(len(train_docs[0]))
pprint(train_docs[0])

# for d in train_docs:
#     pprint(term_exists(d))
    # print(type(term_exists(d)))
    # print(d)
    # print(c)

#
# # 시간 단축을 위한 꼼수로 training corpus의 일부만 사용할 수 있음
#
# tokens = [t for d in train_docs for t in d]

# text = nltk.Text(tokens, name='NMSC')
# selected_words = [f[0] for f in text.vocab().most_common(2000)]
#
# pprint(selected_words)
#
# def term_exists(doc):
#     return {'exists({})'.format(word): (word in set(doc)) for word in selected_words}

train_docs = train_docs[:10]

# for d in train_docs:
#     pprint(term_exists(d))
    # print(type(term_exists(d)))
    # print(d)
    # print(c)

#
# # 시간 단축을 위한 꼼수로 training corpus의 일부만 사용할 수 있음
#
pprint(len(train_docs[0]))
# pprint(train_docs[0])
# train_xy = [(term_exists(d), c) for d, c in train_docs]
# test_xy = [(term_exists(d), c) for d, c in test_docs]
#
# print(train_xy

# classifier = nltk.NaiveBayesClassifier.train(train_xy)