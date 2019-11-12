import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def tokenizer(text):
    """
    форматирование строки - замена пробелов, запятых, приведение к нижнему регистру, разделение на слова по пробелам
    """
    # TODO: improve tokenizer
    text = re.sub('<[^>]*>', '', text)
    text = re.sub(r'[a-z]([A-Z])', r'-\1',
                  text.replace('.', ' . ').replace('/', ' ').replace(',', ' , ')).lower().split()
    return text


def embeddings_download(directory):
    """
    загрузка представлений слов
    :param directory: директория с файлами
    :return: словарь: ключ - слово, значение - векторное представление слова
    """
    embeddings_index = {}
    f = open(os.path.join(directory, 'glove.6B.100d.txt'), encoding='utf-8')

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index


def data_download(review_size, embeddings_index, directory):
    """
    Загрузка данных и их подготовка к подаче в классификатор.
    Берутся первые review_size слов отзыва, если слово отсутствует в словаре, набираем дальше по тексту
    :param review_size: количество используемых слов в отзыве
    :param embeddings_index: словарь embedding'ов
    :param directory: директория с файлами
    :return: обучающая и тестовая выборки
    """
    labels = []
    texts = []
    embed_len = len(embeddings_index[next(iter(embeddings_index))])
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(directory, label_type)
        for file_name in os.listdir(dir_name):
            if file_name[-4:] == '.txt':
                f = open(os.path.join(dir_name, file_name), encoding='utf-8')
                temp_str = f.read()
                temp_str = tokenizer(temp_str)
                review = []
                cnt = 0
                while len(review) != review_size * embed_len and cnt < len(temp_str):
                    element = temp_str[cnt]
                    if element in embeddings_index.keys():
                        review += embeddings_index[element].tolist()
                    cnt += 1
                review_len = len(review)
                if review_len != review_size * embed_len:
                    review += list(np.zeros(review_size * embed_len - review_len))
                texts.append(np.asarray(review))
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)

    doc_feat_len = embed_len * review_size
    docs_count = len(texts)

    X = np.zeros((docs_count, doc_feat_len))
    for i in range(len(texts)):
        X[i] = np.asarray(texts[i])

    y = np.array(labels)
    y = y[np.newaxis].T

    return np.append(X, y, axis=1)


def data_download_mean(embeddings_index, directory):
    """
    Загрузка данных и их подготовка к подаче в классификатор.
    Берем средний вектор всего документа для получения более низкой размерности
    :param embeddings_index: словарь embedding'ов
    :param directory: директория с файлами
    :return: обучающая и тренировочная выборки
    """
    labels = []
    texts = []
    embed_len = len(embeddings_index[next(iter(embeddings_index))])
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(directory, label_type)
        for file_name in os.listdir(dir_name):
            review = np.zeros(embed_len, type, 'f')
            if file_name[-4:] == '.txt':
                f = open(os.path.join(dir_name, file_name), encoding='utf-8')
                temp_str = f.read()
                temp_str = tokenizer(temp_str)
                known_words_cnt = 0
                for element in temp_str:
                    if element in embeddings_index.keys():
                        review += np.asarray(embeddings_index[element].tolist())
                        known_words_cnt += 1
                f.close()

                if known_words_cnt > 0:
                    texts.append(review / float(known_words_cnt))
                    if label_type == 'neg':
                        labels.append(0)
                    else:
                        labels.append(1)

    docs_count = len(texts)

    X = np.zeros((docs_count, embed_len))
    for i in range(len(texts)):
        X[i] = np.asarray(texts[i])

    y = np.array(labels)
    y = y[np.newaxis].T

    return np.append(X, y, axis=1)


# region lemmatization, stop_words...
def data_download_mean_stop_words(embeddings_index, directory):
    """
    Загрузка данных и их подготовка к подаче в классификатор.
    Берем средний вектор всего документа для получения более низкой размерности
    :param embeddings_index: словарь embedding'ов
    :param directory: директория с файлами
    :return: обучающая и тренировочная выборки
    """
    labels = []
    texts = []
    with open("english_stop_words") as f:
        stop_words = f.read().splitlines()

    embed_len = len(embeddings_index[next(iter(embeddings_index))])
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(directory, label_type)
        for file_name in os.listdir(dir_name):
            review = np.zeros(embed_len, type, 'f')
            if file_name[-4:] == '.txt':
                f = open(os.path.join(dir_name, file_name), encoding='utf-8')
                temp_str = f.read()
                temp_str = tokenizer(temp_str)
                known_words_cnt = 0
                for element in temp_str:
                    if element in embeddings_index.keys() and not (element in stop_words):
                        review += np.asarray(embeddings_index[element].tolist())
                        known_words_cnt += 1
                f.close()

                if known_words_cnt > 0:
                    texts.append(review / float(known_words_cnt))
                    if label_type == 'neg':
                        labels.append(0)
                    else:
                        labels.append(1)

    docs_count = len(texts)

    X = np.zeros((docs_count, embed_len))
    for i in range(len(texts)):
        X[i] = np.asarray(texts[i])

    y = np.array(labels)
    y = y[np.newaxis].T

    return np.append(X, y, axis=1)


def data_download_mean_lemmas(embeddings_index, directory):
    """
    Загрузка данных и их подготовка к подаче в классификатор.
    Берем средний вектор всего документа для получения более низкой размерности
    :param embeddings_index: словарь embedding'ов
    :param directory: директория с файлами
    :return: обучающая и тренировочная выборки
    """
    labels = []
    texts = []
    wnl = WordNetLemmatizer()
    embed_len = len(embeddings_index[next(iter(embeddings_index))])
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(directory, label_type)
        for file_name in os.listdir(dir_name):
            review = np.zeros(embed_len, type, 'f')
            if file_name[-4:] == '.txt':
                f = open(os.path.join(dir_name, file_name), encoding='utf-8')
                temp_str = f.read()
                temp_str = tokenizer(temp_str)
                known_words_cnt = 0
                for element in temp_str:
                    if element in embeddings_index.keys():
                        wnl.lemmatize(element)
                        review += np.asarray(embeddings_index[element].tolist())
                        known_words_cnt += 1
                f.close()

                if known_words_cnt > 0:
                    texts.append(review / float(known_words_cnt))
                    if label_type == 'neg':
                        labels.append(0)
                    else:
                        labels.append(1)

    docs_count = len(texts)

    X = np.zeros((docs_count, embed_len))
    for i in range(len(texts)):
        X[i] = np.asarray(texts[i])

    y = np.array(labels)
    y = y[np.newaxis].T

    return np.append(X, y, axis=1)


def data_download_mean_stop_words_lemmas(embeddings_index, directory):
    """
    Загрузка данных и их подготовка к подаче в классификатор.
    Берем средний вектор всего документа для получения более низкой размерности
    :param embeddings_index: словарь embedding'ов
    :param directory: директория с файлами
    :return: обучающая и тренировочная выборки
    """
    labels = []
    texts = []
    wnl = WordNetLemmatizer()
    with open("english_stop_words") as f:
        stop_words = f.read().splitlines()

    embed_len = len(embeddings_index[next(iter(embeddings_index))])
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(directory, label_type)
        for file_name in os.listdir(dir_name):
            review = np.zeros(embed_len, type, 'f')
            if file_name[-4:] == '.txt':
                f = open(os.path.join(dir_name, file_name), encoding='utf-8')
                temp_str = f.read()
                temp_str = tokenizer(temp_str)
                known_words_cnt = 0
                for element in temp_str:
                    if element in embeddings_index.keys() and not (element in stop_words):
                        wnl.lemmatize(element)
                        review += np.asarray(embeddings_index[element].tolist())
                        known_words_cnt += 1
                f.close()

                if known_words_cnt > 0:
                    texts.append(review / float(known_words_cnt))
                    if label_type == 'neg':
                        labels.append(0)
                    else:
                        labels.append(1)

    docs_count = len(texts)

    X = np.zeros((docs_count, embed_len))
    for i in range(len(texts)):
        X[i] = np.asarray(texts[i])

    y = np.array(labels)
    y = y[np.newaxis].T

    return np.append(X, y, axis=1)


# endregion


def dim_reduction_plot(X, y):
    """
    построение трехмерного графика сниженной размерности
    :param X:
    :param y:
    :return:
    """
    fig = plt.figure(1, figsize=(16, 9))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=2).fit_transform(X)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y,
               cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigen vector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigen vector")
    ax.w_yaxis.set_ticklabels([])

    plt.show()
    print("The number of features in the new subspace is ", X_reduced.shape[1])


def dim_reduction_plot_tsne(X, y):
    """
    Стохастическое вложение соседей с t-распределением (англ. t-distributed Stochastic Neighbor Embedding, t-SNE) —
    это алгоритм машинного обучения для визуализации, разработанный Лоренсом ван дер Маатеном и Джеффри Хинтоном.
    Он является техникой нелинейного снижения размерности, хорошо подходящей для вложения данных высокой размерности
    для визуализации в пространство низкой размерности (двух- или трехмерное). В частности, метод моделирует каждый
    объект высокой размерности двух- или трёхмерной точкой таким образом, что похожие объекты моделируются близко
    расположенными точками, а непохожие точки моделируются с большой P точками, далеко друг от друга отстоящими.
    """
    x_reduced = TSNE(n_components=2).fit_transform(X)
    sns.scatterplot(x_reduced[:, 0], x_reduced[:, 1], hue=y, legend='full')


def roc_curve_own(model, X, y):
    """
    построение roc-кривой модели, изменяемый параметр - w_0
    :param model: обученная модель
    :param X: признаки
    :param y: ответы
    :return: график roc-кривой
    """
    roc_array = np.zeros((len(X), 2))  # сверху fpr, снизу tpr
    a_y = np.zeros((len(X), 2))  # сверху decision_function(x_i), снизу метка класса
    m = np.zeros(len(X))

    # сортировка двумерного массива по строке m
    m = model.decision_function(X)
    a_y[:, 0] = np.array(m.T).astype(float)  # np.ravel(m.T)
    a_y[:, 1] = y.T
    l_a_y = list(a_y)
    l_a_y = sorted(l_a_y, key=lambda x: x[0])
    a_y_sorted = np.asarray(l_a_y)

    # заполняем roc_array tpr и fpr
    if a_y_sorted[len(X) - 1, 1] == 0:
        roc_array[len(X) - 1, 0] = 1
    else:
        roc_array[len(X) - 1, 1] = 1

    for i in reversed(range(len(X) - 1)):
        if a_y_sorted[i, 1] == 0:
            roc_array[i, 0] = roc_array[i + 1, 0] + 1
            roc_array[i, 1] = roc_array[i + 1, 1]
        else:
            roc_array[i, 0] = roc_array[i + 1, 0]
            roc_array[i, 1] = roc_array[i + 1, 1] + 1

    y_neg_cnt = len(list(filter(lambda x: (x == 0), y)))
    y_pos_cnt = len(list(filter(lambda x: (x == 1), y)))

    if y_neg_cnt > 0 and y_pos_cnt > 0:
        roc_array[:, 0] = roc_array[:, 0] / float(y_neg_cnt)
        roc_array[:, 1] = roc_array[:, 1] / float(y_pos_cnt)

    plt.plot(roc_array[:, 0], roc_array[:, 1])
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


start_time = time.time()

glove_dir = 'D:\\datasets\\glove.6B'
imdb_dir: str = 'D:\\datasets\\aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
test_dir = os.path.join(imdb_dir, 'test')

# загрузка embedding'ов
embeddings_index = embeddings_download(glove_dir)

# region make_csv
# Xy_train = data_download_mean_lemmas(embeddings_index, train_dir)
# Xy_test = data_download_mean_lemmas(embeddings_index, test_dir)
#
# pd.DataFrame(np.append(Xy_train, Xy_test, axis=0)).to_csv("imdb_mean_stop_lemmas.csv")
# endregion

# загрузка данных, формирование тренировочной и тестовой выборок
imdb_data = np.genfromtxt('imdb_mean.csv', delimiter=',')
X = imdb_data[1:, 1:-1]
y = imdb_data[1:, -1:]

# масштабирование выборок
scaler = StandardScaler().fit_transform(X)

# разделение выборки на тренировочную и тестовую
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

# region двумерный график tsne
# построение графика сниженной размерности
# a = np.append(X_test[:50], X_test[-50:], axis=0)
# b = np.append(y_test[:50], y_test[-50:], axis=0)
# dim_reduction_plot_tsne(X_test, y_test)
# endregion

# region linearsvm
# clf = LinearSVC(penalty='l2', loss='squared_hinge',
#                 dual=False, tol=0.0001, C=1, multi_class='ovr',
#                 fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0
#                 , random_state=0, max_iter=100000)
# clf.fit(X_train, y_train)
#
# print('Accuracy of linear SVC on training set: {:.2f}'.format(clf.score(X_train, y_train) * 100))
# print('Accuracy of linear SVC on test set: {:.2f}'.format(clf.score(X_test, y_test) * 100))
# endregion

# nonlinear svm
clf_SVC = SVC(C=0.1, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True,
              probability=False, tol=0.001, cache_size=1000, class_weight=None,
              verbose=True, max_iter=-1, decision_function_shape="ovr", random_state=0)
clf_SVC.fit(X_train, y_train)
y_pred = clf_SVC.predict(X_test)

print('Accuracy of SVC on training set: {:.2f}'.format(clf_SVC.score(X_train, y_train) * 100))
print('Accuracy of SVC on test set: {:.2f}'.format(clf_SVC.score(X_test, y_test) * 100))

# roc_curve
roc_curve_own(clf_SVC, X_test, y_test)

# confusion_matrix
conf_matrix = confusion_matrix(clf_SVC.predict(X_test), y_test)

total_time = round((time.time() - start_time))
print("Time elapsed: %s minutes %s seconds" % ((total_time // 60), round(total_time % 60)))

# region optimal c parameter
# optimal c parameter
# c = np.logspace(start=-15, stop=1000, base=1.02)
# param_grid = {'C': c}
#
# grid = GridSearchCV(clf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
# grid.fit(X_train, y_train)
#
# print("The best parameters are %s with a score of %0.0f" % (grid.best_params_, grid.best_score_ * 100))
# print("Best estimator accuracy on test set {:.2f} ".format(grid.best_estimator_.score(X_test, y_test) * 100))
# endregion

# TODO: csv plugin --- updated to prof version
# TODO: tf-idf from scikitlearn
# TODO: git ---
# TODO: improve multiclass classification
# TODO: lemmatization - прироста в точности не дает
# TODO: delete stopwords - прироста в точности не дает
# TODO: delete stopwords + lemmatization - прироста в точности не дает
# TODO: new dataset
# TODO: start nn
