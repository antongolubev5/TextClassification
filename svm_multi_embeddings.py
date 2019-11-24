import os
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from nltk import word_tokenize, PorterStemmer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.svm import LinearSVC
from tok import word_tokenize
from nltk.corpus import stopwords


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


def embeddings_download(directory):
    """
    загрузка представлений слов
    :param directory: директория с файлами
    :return: словарь: ключ - слово, значение - embedding
    """
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8')

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

    X_train = np.zeros((docs_count, doc_feat_len))
    for i in range(len(texts)):
        X_train[i] = np.asarray(texts[i])

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
    dict_labels = {}
    cnt = 0
    texts = []
    embed_len = len(embeddings_index[next(iter(embeddings_index))])
    review = np.zeros(embed_len, type, 'f')

    for subdir, dirs, files in os.walk(directory):
        for dir in dirs:
            dict_labels[dir] = cnt
            cnt += 1

    for element in dict_labels:
        for subdir, dirs, files in os.walk(os.path.join(directory, element)):
            for file in files:
                review = np.zeros(embed_len, type, 'f')
                f = open(os.path.join(subdir, file), encoding='utf-8', errors='ignore')
                temp_str = ""
                known_words_cnt = 0
                flag = False
                for line in f.readlines():
                    if line == '\n':
                        flag = True
                    if flag:
                        temp_str = temp_str + " " + line
                temp_str = tokenizer(temp_str)
                for word in temp_str:
                    if word in embeddings_index.keys():
                        review += np.asarray(embeddings_index[word].tolist())
                        known_words_cnt += 1
                if known_words_cnt > 0:
                    texts.append(review / float(known_words_cnt))
                    labels.append(dict_labels[element])
                f.close()

    docs_count = len(texts)
    X = np.zeros((docs_count, embed_len))

    for i in range(docs_count):
        X[i] = np.asarray(texts[i])

    y = np.array(labels)
    y = y[np.newaxis].T

    return np.append(X, y, axis=1)


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


if __name__ == "__main__":

    start_time = time.time()

    if 'DESKTOP-TF87PFA' in os.environ['COMPUTERNAME']:
        glove_dir = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\glove.6B'
        imdb_dir: str = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\aclImdb'
        train_dir = os.path.join(imdb_dir, 'train')
        test_dir = os.path.join(imdb_dir, 'test')
        reuters_mean_csv = 'твой путь'
        to_reuters_mean_csv = 'твой путь'
    else:
        glove_dir = 'D:\\datasets\\glove.6B'
        imdb_dir: str = 'D:\\datasets\\aclImdb'
        train_dir = os.path.join(imdb_dir, 'train')
        test_dir = os.path.join(imdb_dir, 'test')
        reuters_mean_csv = 'D:\\datasets\\csv_files\\reuters_mean.csv'
        to_reuters_mean_csv = 'D:\\datasets\\csv_files'

    stop_words = set(stopwords.words('english'))
    labels_cnt = 90

    # загрузка embedding'ов
    embeddings_index = embeddings_download(glove_dir)
    reuters = data_download_mean_reuters(embeddings_index)
    # pd.DataFrame(reuters).to_csv(to_reuters_mean_csv)

    # загрузка данных, формирование тренировочной и тестовой выборок
    data = np.genfromtxt(reuters_mean_csv, delimiter=',')
    X = data[1:, 1:-labels_cnt]
    y = data[1:, -labels_cnt:]

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

    # answers = clf_SVC.predict(X_test)
    # results = np.zeros(90)
    # difference = 0
    # for i in range(answers.shape[0]):
    #     for j in range(len(answers[0])):
    #         if answers[i][j] != y_test[i][j]:
    #             difference += 1
    #
    # print((answers.shape[0]*answers.shape[1]-2091)/(answers.shape[0]*answers.shape[1]))
    #
    # for i in range(90):
    #     print("Точность предсказания", i, "-го класса = ", accuracy_score(y_test[:, i], clf_SVC.predict(X_test)[:, i]))
    #     results[i] = (accuracy_score(y_test[:, i], clf_SVC.predict(X_test)[:, i]))
    #
    # print("Точность предсказания в среднем по классам = ", results.mean())

    print('Accuracy of SVC on training set: {:.2f}'.format(clf_SVC.score(X_train, y_train) * 100))
    print('Accuracy of SVC on test set: {:.2f}'.format(clf_SVC.score(X_test, y_test) * 100))

    total_time = round((time.time() - start_time))
    print("Time elapsed: %s minutes %s seconds" % ((total_time // 60), round(total_time % 60)))
