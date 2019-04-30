


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


tokens = [t for d in [ast.literal_eval(sentence) for sentence in train_sentence_docs] for t in d]
text = nltk.Text(tokens, name='NMSC')

# 시간 단축을 위한 꼼수로 training corpus의 일부만 사용할 수 있음
selected_words = [f[0] for f in text.vocab().most_common(2000)]

def term_exists(doc):
    return {'exists({})'.format(word): (word in set(doc)) for word in selected_words}

train_docs = train_docs[:1000]


train_xy = [(term_exists(d), c) for d, c in train_docs]
test_xy = [(term_exists(d), c) for d, c in test_docs]

classifier = nltk.NaiveBayesClassifier.train(train_xy)
print(nltk.classify.accuracy(classifier, test_xy))
# => 0.75442
classifier.show_most_informative_features(10)
# Most Informative Features
#         exists(쓰레기/Noun) = True                0 : 1      =     11.9 : 1.0
#          exists(인생/Noun) = True                1 : 0      =     10.0 : 1.0
#          exists(최고/Noun) = True                1 : 0      =      9.5 : 1.0
#    exists(괜찮다/Adjective) = True                1 : 0      =      8.6 : 1.0
#   exists(재미없다/Adjective) = True                0 : 1      =      8.2 : 1.0
# exists(ㅡㅡ/KoreanParticle) = True                0 : 1      =      8.1 : 1.0
#    exists(재밌다/Adjective) = True                1 : 0      =      7.6 : 1.0
#    exists(아깝다/Adjective) = True                0 : 1      =      7.6 : 1.0
#   exists(지루하다/Adjective) = True                0 : 1      =      7.6 : 1.0
#    exists(슬프다/Adjective) = True                1 : 0      =      7.2 : 1.0