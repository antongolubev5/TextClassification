from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def tsne_plot(labels, tokens, classes, clusters):
    tsne_model = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=33)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []

    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    colors = cm.rainbow(np.linspace(0, 1, clusters))
    plt.figure(figsize=(16, 9))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=colors[classes[i]], alpha=0.75)
        plt.annotate(labels[i], alpha=0.75, xy=(x[i], y[i]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom', size=10)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # загрузка обученной модели
    model = Word2Vec.load('/media/anton/ssd2/data/datasets/языковые модели/habr_w2v/tweets_model.w2v')

    labels = []
    tokens = []
    classes = []

    samples = 15
    for i, word in enumerate(
            ['математика', 'python', 'искусство', 'машина', 'кино', 'iphone', 'воздух', 'природа', 'любить']):
        tokens.append(model.wv[word])
        labels.append(word)
        classes.append(i)
        for similar_word, similarity in model.wv.most_similar(word, topn=20):
            tokens.append(model.wv[similar_word])
            labels.append(similar_word)
            classes.append(i)

    tsne_plot(labels, tokens, classes, samples)
