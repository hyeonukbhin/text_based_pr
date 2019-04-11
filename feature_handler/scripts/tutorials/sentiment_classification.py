def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]  # header 제외
    return data


train_data = read_data('../ratings_train.txt')
test_data = read_data('../ratings_test.txt')

# comp_train_data = train_data[:150]
# comp_test_data = test_data[:50]
comp_train_data = train_data[:]
comp_test_data = test_data[:]

# row, column의 수가 제대로 읽혔는지 확인
print(len(comp_train_data))  # nrows: 150000
print(len(comp_train_data[0]))  # ncols: 3
print(len(comp_test_data))  # nrows: 50000
print(len(comp_test_data[0]))  # ncols: 3

from konlpy.tag import Okt

pos_tagger = Okt()

def tokenize(doc):
    # norm, stem은 optional
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]


train_docs = [(tokenize(row[1]), row[2]) for row in comp_train_data]
test_docs = [(tokenize(row[1]), row[2]) for row in comp_test_data]
# 잘 들어갔는지 확인
from pprint import pprint

pprint(train_docs[0])
# => [(['아/Exclamation',
#   '더빙/Noun',
#   '../Punctuation',
#   '진짜/Noun',
#   '짜증/Noun',
#   '나다/Verb',
#   '목소리/Noun'],
#  '0')]
print(type(train_docs))

import pandas as pd

df_train = pd.DataFrame(train_docs)
df_train.columns = ["sentence", "ratings"]

df_test = pd.DataFrame(test_docs)
df_test.columns = ["sentence", "ratings"]

def save_df(df, filename):
    df.to_csv(filename,  mode="w", sep=',')


def read_df(filename):
    df = pd.read_csv(filename, sep=',', na_values=".", index_col=0)
    return df

save_df(df_train, "train_docs.csv")
save_df(df_test, "test_docs.csv")
