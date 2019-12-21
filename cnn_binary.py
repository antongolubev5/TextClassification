import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import layers, models, regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.datasets import imdb
from mpl_toolkits.mplot3d import Axes3D
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tok import word_tokenize
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences


def tokenizer(text):
    """
    форматирование строки https://github.com/kootenpv/tok/blob/master/README.md
    возвращает предложение как список
    """
    result = word_tokenize(text)
    drop = [element.lower() for element in result if not (element in stop_words)]

    return drop


def tokenizer_tfidf(text):
    """
    форматирование строки https://github.com/kootenpv/tok/blob/master/README.md
    возвращает предложение как строку
    """
    ps = PorterStemmer()
    result = word_tokenize(text)
    drop = [element.lower() for element in result if not (element in stop_words)]

    return ' '.join(drop)


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
                temp_str = Tokenizer(temp_str)
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
    df = pd.DataFrame(columns=['text', 'label'])
    labels = []
    texts = []
    embed_len = len(embeddings_index[next(iter(embeddings_index))])
    num = 0
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
                num += 1

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


def glove_representation(glove_dir):
    """
    перенос glove-представлений слов из txt файла в словарь
    :param glove_dir:
    :return:
    """
    embeddings_dict = {}

    with open(glove_dir + '\\glove.6B.100d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    return embeddings_dict


def word_vectorizing_keras(csv_file, max_features, max_len):
    """
    векторизация текстов с помощью встроенного keras-функционала
    :param csv_file: датасет
    :param max_features: размер алфавита
    :param max_len: макс длина вектора-текста
    :return:
    """
    labels = np.asarray(list(csv_file['1'])).astype('float32')
    texts = list(csv_file['0'])

    X = [one_hot(text, max_features) for text in texts]
    X = pad_sequences(X, maxlen=max_len)

    return X, labels


def word_vectorizing(csv_file, embeddings_dict, maxlen):
    """
    векторизация текстов с помощью glove representations
    :param csv_file: датасет
    :param embeddings_dict: словарь представления слов
    :param maxlen: фиксированная длина отзыва
    :return:
    """
    glove_len = len(list(embeddings_dict.values())[0])

    texts = list(csv_file['0'])
    texts = [tokenizer(text) for text in texts]

    labels = np.asarray(list(csv_file['1'])).astype('float32')

    representations = np.zeros((len(texts), maxlen, glove_len), dtype='float32')

    for i in range((len(texts))):
        for j in range(min(len(texts[i]), maxlen)):
            if texts[i][j] in embeddings_dict.keys():
                representations[i][j] = embeddings_dict[texts[i][j]]
            else:
                representations[i][j] = np.zeros(100)

    return representations, labels


def build_model_cnn(embed_len):
    """
    построение модели cnn
    :param embed_len: длина векторного представления
    :return:
    """
    model = models.Sequential()
    # model.add(layers.SimpleRNN(embed_len, return_sequences=True))
    # model.add(layers.SimpleRNN(embed_len, return_sequences=True))
    model.add(layers.SimpleRNN(embed_len))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_model_cnn_with_embedding(max_features, embed_len):
    """
    построение модели cnn со слоем embedding
    :param max_features: кол во слов в алфавите
    :param embed_len: длина векторного представления
    :return:
    """
    model = models.Sequential()
    model.add(layers.Embedding(max_features, embed_len, input_length=max_len))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def loss_graph(model_history):
    """
    график потерь на этапах обучения и проверки
    :return:
    """
    history_dict = model_history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(model_history.epoch) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def accuracy_graph(model_history):
    """
    график точности на этапах обучения и проверки
    :return:
    """
    history_dict = model_history.history
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    epochs = range(1, len(model_history.epoch) + 1)
    plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    start_time = time.time()
    stop_words = set(stopwords.words('english'))

    if 'DESKTOP-TF87PFA' in os.environ['COMPUTERNAME']:
        glove_dir = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\glove.6B'
        imdb_dir: str = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\aclImdb'
        train_dir = os.path.join(imdb_dir, 'train')
        test_dir = os.path.join(imdb_dir, 'test')
        imdb_csv = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\csv_files\\imdb.csv'
        to_imdb_csv = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\csv_files'
        X_path = 'D:\\datasets\\npy\\X.npy'
        y_path = 'D:\\datasets\\npy\\y.npy'

    else:
        glove_dir = 'D:\\datasets\\glove.6B'
        imdb_dir: str = 'D:\\datasets\\aclImdb'
        train_dir = os.path.join(imdb_dir, 'train')
        test_dir = os.path.join(imdb_dir, 'test')
        imdb_csv = 'D:\\datasets\\csv_files\\imdb.csv'
        to_imdb_csv = 'D:\\datasets\\csv_files'
        X_path = 'D:\\datasets\\npy\\X.npy'
        y_path = 'D:\\datasets\\npy\\y.npy'

    # region make_csv
    # Xy_train = csv_from_txts(train_dir)
    # Xy_test = csv_from_txts(test_dir)
    # pd.DataFrame(np.append(Xy_train, Xy_test, axis=0)).to_csv(to_imdb_mean_csv)
    # endregion

    # загрузка данных, векторизация текстов
    max_features = 10000
    max_len = 500
    embed_len = 32

    imdb_data = pd.read_csv(imdb_csv)
    embeddings_dict = glove_representation(glove_dir)
    # X, y = word_vectorizing(imdb_data, embeddings_dict, max_len)
    X, y = word_vectorizing_keras(imdb_data, max_features, max_len)

    # np.save("X", X)
    # np.save("y", y)
    # X = np.load(X_path)
    # y = np.load(y_path)

    # масштабирование выборок
    # Scaler = StandardScaler().fit_transform(X)

    # разделение выборки на тренировочную, тестовую и валидационную
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
    val_size = round(len(X_train) * 0.33)
    X_val = X_train[:val_size]
    X_train = X_train[val_size:]
    y_val = y_train[:val_size]
    y_train = y_train[val_size:]

    # (max_words, embedding_dim, embedding_matrix, maxlen)
    mdl = build_model_cnn_with_embedding(max_features, embed_len)

    history = mdl.fit(X_train,
                      y_train,
                      epochs=2,
                      batch_size=128,
                      validation_split=0.2)
    loss_graph(history)
    accuracy_graph(history)
    print(mdl.evaluate(X_test, y_test))
    print(mdl.summary())

    # my_own_text = ["test"]
    # my_own_test = vector_mdl.transform(my_own_text)
    # print(mdl.predict(my_own_test))

    total_time = round((time.time() - start_time))
    print("Time elapsed: %s minutes %s seconds" % ((total_time // 60), round(total_time % 60)))
