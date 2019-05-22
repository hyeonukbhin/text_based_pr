# -*- coding: utf-8 -*-
from pprintpp import pprint
import gensim
import pandas as pd
from sklearn.manifold import TSNE

from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim
import gensim.models as g



def read_df(filename):
    df = pd.read_csv(filename, sep=',', na_values=".", index_col=0)
    return df






def main():


    model = gensim.models.Doc2Vec.load("personality_doc2vec.model")

    # 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
    mpl.rcParams['axes.unicode_minus'] = False


    vocab = list(model.wv.vocab)
    X = model[vocab]

    print(len(X))
    print(X[0][:10])
    tsne = TSNE(n_components=2)

    # 100개의 단어에 대해서만 시각화
    X_tsne = tsne.fit_transform(X[:100, :])
    # X_tsne = tsne.fit_transform(X)

    df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])
    # df.shape

    fig = plt.figure()
    fig.set_size_inches(40, 20)
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df['x'], df['y'])

    for word, pos in df.iterrows():

        ax.annotate(word, pos, fontsize=20)
    plt.show()


main()
