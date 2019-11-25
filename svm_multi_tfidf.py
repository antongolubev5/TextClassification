import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.utils.np_utils import to_categorical
from mpl_toolkits.mplot3d import Axes3D
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.svm import LinearSVC
from tok import word_tokenize


def tokenizer(text):
    """
    форматирование строки https://github.com/kootenpv/tok/blob/master/README.md
    regexp, stop_words, lowercase, stemmer
    """

    ps = PorterStemmer()
    result = word_tokenize(text)
    output = []

    for element in result:
        if not (element in stop_words):
            output.append(ps.stem(element))

    return output


def data_download_mean_reuters(embeddings_index):
    """
    Загрузка данных и их подготовка к подаче в классификатор.
    Берем средний вектор всего документа для получения более низкой размерности
    :param embeddings_index: словарь embedding'ов
    :param directory: директория с файлами
    :return: обучающая и тренировочная выборки
    """
    labels = []
    dict_labels = {}
    cnt = 0
    texts = []
    cats = []
    labels_marks = []
    flattened_cats = []
    table_marks = {}
    embed_len = len(embeddings_index[next(iter(embeddings_index))])

    # словарь категорий
    f = open("D:\\datasets\\reuters\\cats.txt", 'r')
    for line in f:
        dict_labels[line.strip().split(" ")[0].split('/')[1]] = line.strip().split(' ')[1:]
        cats.append(line.strip().split(' ')[1:])

    # flattening
    for x in cats:
        for y in x:
            flattened_cats.append(y)

    cats = np.unique(np.asarray(flattened_cats))

    i = 1
    for element in cats:
        table_marks[element] = i
        i += 1

    # стоп слова
    stop_words_path = "D:\\datasets\\reuters\\stopwords"
    stop_words = []
    f = open(stop_words_path, encoding='utf-8', errors='ignore')
    for line in f.readlines():
        stop_words.append(line.strip())

    # обрабатываем все файлы из директории
    file_path = "D:\\datasets\\reuters\\texts"

    for element in dict_labels.keys():
        review = np.zeros(embed_len, type, 'f')
        f = open(os.path.join(file_path, element), encoding='utf-8', errors='ignore')
        temp_str = ""
        known_words_cnt = 0
        temp_str = f.read()
        temp_str = word_tokenize(temp_str)
        for word in temp_str:
            if word in embeddings_index.keys() and not (word in stop_words):
                review += np.asarray(embeddings_index[word].tolist())
                known_words_cnt += 1
        if known_words_cnt > 0:
            texts.append(review / float(known_words_cnt))
            labels.append(dict_labels[element])
        f.close()

    # кодировка признаков по таблице
    i = 0

    for element in labels:
        labels_marks.append([])
        for subelement in element:
            labels_marks[i].append(table_marks[subelement])
        i += 1

    docs_count = len(texts)
    X = np.zeros((docs_count, embed_len))

    for i in range(docs_count):
        X[i] = np.asarray(texts[i])

    mlb = MultiLabelBinarizer()
    labels_marks = mlb.fit_transform(labels_marks)

    return np.append(X, labels_marks, axis=1)


def dim_reduction_plot(X, y):
    fig = plt.figure(1, figsize=(16, 9))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=2).fit_transform(X)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y,
               cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
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


def csv_from_txts(directory):
    """
    Загрузка данных в csv file.
    Берем средний вектор всего документа для получения более низкой размерности
    :param directory: директория с файлами
    :return: обучающая и тренировочная выборки
    """
    df = pd.DataFrame(columns=['text', 'label'])
    dict_labels = {}
    cats = []
    labels = []
    table_marks = {}
    labels_marks = []
    flattened_cats = []

    # словарь категорий
    f = open("D:\\datasets\\reuters\\cats.txt", 'r')
    for line in f:
        dict_labels[line.strip().split(" ")[0].split('/')[1]] = line.strip().split(' ')[1:]
        cats.append(line.strip().split(' ')[1:])

    # flattening
    for x in cats:
        for y in x:
            flattened_cats.append(y)

    cats = np.unique(np.asarray(flattened_cats))

    i = 1
    for element in cats:
        table_marks[element] = i
        i += 1

    # обрабатываем все файлы из директории
    num = 0
    for element in dict_labels.keys():
        f = open(os.path.join(directory, element), encoding='utf-8', errors='ignore')
        df.loc[num] = [f.read(), 0]
        f.close()
        labels.append(dict_labels[element])
        num += 1

    i = 0
    for element in labels:
        labels_marks.append([])
        for subelement in element:
            labels_marks[i].append(table_marks[subelement])
        i += 1

    df['label'] = labels_marks

    return df


def tf_idf_representation(csv_file):
    """
    векторизация текстов с помощью tf-idf
    :param csv_file:
    :return:
    """
    texts = list(csv_file['text'])
    labels = list(csv_file['label'])

    for i in range(len(labels)):
        labels[i] = labels[i].strip('][').split(', ')

    corpus = list(map(tokenizer, texts))

    for i in range(len(corpus)):
        corpus[i] = " ".join(corpus[i])

    labels = to_categorical(labels)
    # mlb = MultiLabelBinarizer()
    # labels = mlb.fit_transform(labels)
    # corpus = list(texts)

    vectorizer = TfidfVectorizer(decode_error='ignore', lowercase=True, stop_words='english',
                                 min_df=0.0001)
    representations = vectorizer.fit_transform(corpus).toarray()

    return representations, labels, vectorizer


if __name__ == "__main__":

    start_time = time.time()

    if 'DESKTOP-TF87PFA' in os.environ['COMPUTERNAME']:
        glove_dir = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\glove.6B'
        imdb_dir: str = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\aclImdb'
        train_dir = os.path.join(imdb_dir, 'train')
        test_dir = os.path.join(imdb_dir, 'test')
        data_csv = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\csv_files\\newsgroups.csv'
        save_data_csv = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\csv_files'

    else:
        imdb_dir: str = 'D:\\datasets\\aclImdb'
        train_dir = os.path.join(imdb_dir, 'train')
        test_dir = os.path.join(imdb_dir, 'test')
        data_csv = 'D:\\datasets\\csv_files\\newsgroups.csv'
        save_data_csv = 'D:\\datasets\\csv_files'

    stop_words = set(stopwords.words('english'))

    # region make csv file
    # reuters = csv_from_txts(reuters_dir)
    # pd.DataFrame(reuters).to_csv(save_data_csv)
    # endregion

    # загрузка данных, формирование тренировочной и тестовой выборок
    data = pd.read_csv(data_csv)
    X, y, vector_mdl = tf_idf_representation(data)

    # масштабирование выборки
    scaler = StandardScaler()
    scaler.fit_transform(X)

    # разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

    # построение трехмерного графика сниженной размерности
    # dim_reduction_plot_tsne(X_test, y_test)

    # nonlinear svm
    clf_SVC = OneVsRestClassifier(LinearSVC())
    # clf_SVC = SVC(C=0.1, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True,
    #               probability=False, tol=0.001, cache_size=1000, class_weight=None,
    #               verbose=0, max_iter=-1, decision_function_shape="ovr", random_state=0)
    clf_SVC.fit(X_train, y_train)

    print('Accuracy of SVC on training set: {:.2f}'.format(clf_SVC.score(X_train, y_train) * 100))
    print('Accuracy of SVC on test set: {:.2f}'.format(clf_SVC.score(X_test, y_test) * 100))

    total_time = round((time.time() - start_time))
    print("Time elapsed: %s minutes %s seconds" % ((total_time // 60), round(total_time % 60)))
