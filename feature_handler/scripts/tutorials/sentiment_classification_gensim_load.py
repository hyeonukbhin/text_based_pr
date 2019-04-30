from collections import namedtuple
import pandas as pd
import nltk
import ast
from pprintpp import pprint


import gensim

model = gensim.models.Doc2Vec.load("doc2vec.model")

pprint(model.most_similar('공포/Noun'))

pprint(model.most_similar('ㅋㅋ/KoreanParticle'))
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


pprint(model.most_similar(positive=['여자/Noun', '왕/Noun'], negative=['남자/Noun']))
# => [('악당/Noun', 0.32974398136138916),
#     ('곽지민/Noun', 0.305545836687088),
#     ('심영/Noun', 0.2899821400642395),
#     ('오빠/Noun', 0.2856029272079468),
#     ('전작/Noun', 0.2840743064880371),
#     ('눈썹/Noun', 0.28247544169425964),
#     ('광팬/Noun', 0.2795347571372986),
#     ('지능/Noun', 0.2794691324234009),
#     ('박보영/Noun', 0.27567577362060547),
#     ('강예원/Noun', 0.2734225392341614)]

pprint(model.most_similar('왕/Noun'))


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
tagged_test_docs = [TaggedDocument(d, [c]) for d, c in test_docs]

pprint(tagged_train_docs)

train_x = [model.infer_vector(doc.words) for doc in tagged_train_docs]
train_y = [doc.tags[0] for doc in tagged_train_docs]
len(train_x)       # 사실 이 때문에 앞의 term existance와는 공평한 비교는 아닐 수 있다
# => 150000
len(train_x[0])
# => 300
test_x = [model.infer_vector(doc.words) for doc in tagged_test_docs]
test_y = [doc.tags[0] for doc in tagged_test_docs]
len(test_x)
# => 50000
len(test_x[0])
# => 300

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=1234)
classifier.fit(train_x, train_y)
result = classifier.score(test_x, test_y)
# classifier.
print(result)
# => 0.78246000000000004



# text.concordance('왕/Noun', lines=10)
# # => Displaying 10 of 145 matches:
# #   Josa 로맨스/Noun 냐/Josa ,,/Punctuation 왕/Noun 짜증/Noun ...../Punctuation 아주/Noun 전
# #   /Noun 함/Noun ../Punctuation 결말/Noun 왕/Noun 실망/Noun 임/Noun 전작/Noun 에/Josa 비/Nou
# #   nction 얼굴/Noun 만/Josa 예쁘다/Adjective 왕/Noun 되다/Verb 맞다/Verb 드라마/Noun 라도/Josa 도덕
# #   /Noun 스릴러/Noun 임/Noun ?/Punctuation 왕/Noun 실망/Noun ./Punctuation 연기/Noun 대본/No
# #   b 금/Noun 사인방/Noun ㅠㅠ/KoreanParticle 왕/Noun 잼/Noun 없다/Adjective ./Punctuation 정
# #   osa 서유기/Noun 보다/Josa 희극/Noun 지/Josa 왕/Noun 이/Josa 더/Noun 최고/Noun 라/Josa 생각/Nou
# #   접/Noun 한/Josa 걸작/Noun ./Punctuation 왕/Noun ,/Punctuation 너무/Noun 감동/Noun 적/Suf
# #   Josa 온/Noun 거/Noun 처럼/Noun 제나라/Noun 왕/Noun 과/Josa 군사/Noun 들/Suffix 을/Josa 속이다/
# #   다/Verb ./Punctuation 기대하다/Adjective 왕/Noun 지루/Noun .../Punctuation 제니퍼/Noun 틸리
# #   tive 움/Noun 짜증/Noun .../Punctuation 왕/Noun 짜증/Noun ../Punctuation 사람/Noun 마다/J


# 텍스트의 다양한 표현법에 대해 살펴보았다.
# KoNLPy로 데이터를 전처리하고, NLTK로 데이터를 탐색하고, Gensim으로 문서 벡터를 구해봤다.
# Term existance와 document vector로 센티멘트를 분류해봤다.
# 여기서는 영화 리뷰에 대한 센티멘트 분석을 했지만 얼마든지 다른 task에도 apply할 수 있음!
# 국회 의안 통과/폐기 예측 *
# 주가 상승/하락 예측
# 나는 어떤 태스크에 적용해볼 수 있을지 생각해보자!