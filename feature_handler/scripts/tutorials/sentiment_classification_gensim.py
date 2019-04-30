from collections import namedtuple
import pandas as pd
import nltk
import ast
from pprintpp import pprint

def read_df(filename):
    df = pd.read_csv(filename, sep=',', na_values=".", index_col=0)
    return df


df_train_docs = read_df("train_docs.csv")
train_sentence_docs = df_train_docs["sentence"].to_list()
train_rating_docs = df_train_docs["ratings"].to_list()

train_docs = [(ast.literal_eval(sentence), train_rating_docs[i]) for i, sentence in enumerate(train_sentence_docs)]

df_test_docs = read_df("test_docs.csv")
test_sentence_docs = df_test_docs["sentence"].to_list()
test_rating_docs = df_test_docs["ratings"].to_list()
test_docs = [(ast.literal_eval(sentence), test_rating_docs[i]) for i, sentence in enumerate(test_sentence_docs)]


TaggedDocument = namedtuple('TaggedDocument', 'words tags')
# 여기서는 15만개 training documents 전부 사용함
tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_docs]

pprint(tagged_train_docs)

tagged_test_docs = [TaggedDocument(d, [c]) for d, c in test_docs]

# pprint(tagged_train_docs[:10])


import gensim
# from gensim.models import doc2vec
# 사전 구축

model = gensim.models.Doc2Vec(vector_size=300, window=10, min_count=5, workers=11, alpha=0.025, min_alpha=0.025, epochs=20)
model.build_vocab(tagged_train_docs)

pprint(tagged_train_docs)

model.train(tagged_train_docs, epochs=model.epochs, total_examples=model.corpus_count)

# doc_vectorizer = doc2vec.Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, seed=1234, total_examples=doc_vectorizer.corpus_count)
# doc_vectorizer.build_vocab(tagged_train_docs)
# # Train document vectors!
# for epoch in range(10):
#     doc_vectorizer.train(tagged_train_docs)
#     doc_vectorizer.alpha -= 0.002  # decrease the learning rate
#     doc_vectorizer.min_alpha = doc_vectorizer.alpha  # fix the learning rate, no decay
# To save

model.save("doc2vec.model")
# doc_vectorizer.save('doc2vec.model')

# pprint(model.most_similar('공포/Noun'))
#
# pprint(model.most_similar('ㅋㅋ/KoreanParticle'))
# => [('ㅎㅎ/KoreanParticle', 0.5768033862113953),
#     ('ㅋ/KoreanParticle', 0.4822016954421997),
#     ('!!!!/Punctuation', 0.4395076632499695),
#     ('!!!/Punctuation', 0.4077949523925781),
#     ('!!/Punctuation', 0.4026390314102173),
#     ('~/Punctuation', 0.40038347244262695),
#     ('~~/Punctuation', 0.39946430921554565),
#     ('!/Punctuation', 0.3899948000907898),
#     ('^^/Punctuation', 0.3852730989456177),
#     ('~^^/Punctuation', 0.3797937035560608)]