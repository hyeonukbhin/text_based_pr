from collections import namedtuple
import pandas as pd
import nltk
import ast
from pprintpp import pprint

def read_df(filename):
    df = pd.read_csv(filename, sep=',', na_values=".", index_col=0)
    return df

TaggedDocument = namedtuple('TaggedDocument', 'words tags')
# 여기서는 15만개 training documents 전부 사용함


df_train_docs = read_df("train_docs.csv")
df_test_docs = read_df("test_docs.csv")

train_docs = df_train_docs["sentence"].to_list()
train_docs = [ast.literal_eval(t) for t in train_docs]


# 형태소 분류
train_docs = [(tokenize(row[1]), row[2]) for row in train_data[1:]]
test_docs = [(tokenize(row[1]), row[2]) for row in test_data[1:]]

#Training data의 token 모으기
tokens = [t for d in train_docs for t in d[0]]
print(len(tokens))

# Load tokens with nltk.Text()
text = nltk.Text(tokens, name='NMSC')
print(text.vocab().most_common(10))

# 텍스트간의 연어 빈번하게 등장하는 단어 구하기
text.collocations()


# term이 존재하는지에 따라서 문서를 분류
selected_words = [f[0] for f in text.vocab().most_common(2000)] # 여기서는 최빈도 단어 2000개를 피쳐로 사용
train_docs = train_docs[:10000] # 시간 단축을 위한 꼼수로 training corpus의 일부만 사용할 수 있음
train_xy = [(term_exists(d), c) for d, c in train_docs]
test_xy = [(term_exists(d), c) for d, c in test_docs]

# nltk의 NaiveBayesClassifier으로 데이터를 트래이닝 시키고, test 데이터로 확인
classifier = nltk.NaiveBayesClassifier.train(train_xy) #Naive Bayes classifier 적용
print(nltk.classify.accuracy(classifier, test_xy))
# => 0.80418

classifier.show_most_informative_features(10)














for a, b in train_docs:
    print(a)
    print(b)
# print(train_docs[1][0])
# print(type(train_docs[1][0]))

test_docs = df_test_docs["sentence"].to_list()
# test_docs = [ast.literal_eval(t) for t in test_docs]


tagged_train_docs = [TaggedDocument(d, c) for d, c in train_docs]
tagged_test_docs = [TaggedDocument(d, [c]) for d, c in test_docs]

from gensim.models import doc2vec
# 사전 구축
doc_vectorizer = doc2vec.Doc2Vec(size=300, alpha=0.025, min_alpha=0.025, seed=1234)
doc_vectorizer.build_vocab(tagged_train_docs)
# Train document vectors!
for epoch in range(10):
    doc_vectorizer.train(tagged_train_docs)
    doc_vectorizer.alpha -= 0.002  # decrease the learning rate
    doc_vectorizer.min_alpha = doc_vectorizer.alpha  # fix the learning rate, no decay
# To save
doc_vectorizer.save('doc2vec.model')

pprint(doc_vectorizer.most_similar('공포/Noun'))
