

import pandas as pd
import ast
import pprint

def read_df(filename):
    df = pd.read_csv(filename, sep=',', na_values=".", index_col=0)
    return df


df_train_docs = read_df("train_docs.csv")
df_test_docs = read_df("test_docs.csv")
train_docs = df_train_docs["sentence"].to_list()
train_docs = [ast.literal_eval(t) for t in train_docs]
print(train_docs)
tokens = [t for d in train_docs for t in d]


print(tokens[0])
print(len(tokens))

import nltk

text = nltk.Text(tokens, name='NMSC')

print(text)
print(len(text.tokens))                 # returns number of tokens
# => 2194536
print(len(set(text.tokens)))            # returns number of unique tokens
# => 48765
from pprintpp import pprint

pprint(text.vocab().most_common(10))    # returns frequency distribution
# => [('./Punctuation', 68630),
#     ('영화/Noun', 51365),
#     ('하다/Verb', 50281),
#     ('이/Josa', 39123),
#     ('보다/Verb', 34764),
#     ('의/Josa', 30480),
#     ('../Punctuation', 29055),
#     ('에/Josa', 27108),
#     ('가/Josa', 26696),
#     ('을/Josa', 23481)]


from matplotlib import font_manager, rc
font_fname = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'     # A font of your choice
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)


text.plot(50)
# train_docs = df_train_docs["sentence"].as_matrix()
# print(len(train_docs))
# tokens = train_docs.flatten()
# print(len(tokens))
# print(tokens)
# # print(type(train_docs[0]).split())
# train_sentence = ast.literal_eval(train_docs)
#
# print(train_sentence)
# # print(fruits[2])
#
# # df_train_docs
#
# tokens = [t for d in train_docs for t in ast.literal_eval(d)]
# print(tokens[0])
# # print(len(tokens))

