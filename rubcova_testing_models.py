import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers, models, Sequential
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tok import word_tokenize
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec


def tokenizer(text):
    """
    форматирование строки:
    приведение к нижнему регистру;
    удаление интернет-ссылок и упоминания имен пользователей через @
    удаление знаков пунктуации;
    удаление стоп слов;
    токенизация (встроенный nltk)
    """

    # ps = PorterStemmer()
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('@[^\s]+', 'USER', text)
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)

    result = word_tokenize(text)

    drop = [element.lower() for element in result] #if not (element in stop_words) and len(element) > 1]

    return drop


def embeddings_download(file_path):
    """
    загрузка представлений слов русскоязычной модели с rusvectores
    :param file_path: директория с файлами
    :return: словарь: ключ - слово, значение - векторное представление слова
    """
    embeddings_index = {}
    f = open(file_path, 'r', encoding='utf-8')

    for line in f:
        values = line.split()
        word = values[0].split("_")[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coeffs
    f.close()

    return embeddings_index


def data_download(file_path):
    """
    загрузка текстов и меток
    :param file_path: директория с двумя csv-файлами
    :return: список raw текстов и список меток
    """
    data_positive = pd.read_csv(file_path + '\\positive.csv', sep=';', encoding='utf-8', header=None)
    data_negative = pd.read_csv(file_path + '\\negative.csv', sep=';', encoding='utf-8', header=None)
    corpus = pd.concat((data_positive, data_negative), axis=0)

    texts = list(corpus[3])
    labels = [1]*len(data_positive) + [0]*len(data_negative)

    return texts, labels


def text_vectorization(texts, labels, embeddings_index):
    """
    векторизация текстов с помощью предварительно обученных embeddings
    :param labels: метки текстов
    :param texts: тексты
    :param embeddings_index: предварительно обученные embeddings
    :return:
    """
    # tokenization
    # texts = [tokenizer(text) for text in texts]

    # распределение длин текстов
    # plt.style.use('ggplot')
    # # plt.figure(figsize=(16, 9))
    # # facecolor='g'
    # n, bins, patches = plt.hist([len(text) for text in texts], 50, density=True)
    # plt.xlabel('Number of words in a text')
    # plt.ylabel('Share of texts')
    # plt.axis([0, 40, 0, 0.12])
    # plt.grid(True)
    # plt.show()

    embed_len = len(embeddings_index['лес'])
    text_len = 12

    y = np.asarray(labels)
    X = np.zeros((len(texts), text_len, embed_len), dtype=np.float16)

    for i in range(len(texts)):
        for j in range(min(len(texts[i]), text_len)):
            # if texts[i][j] in embeddings_index.keys():
            X[i][j] = embeddings_index[texts[i][j]]

    return X, y


def nonlinear_svm(X_train, X_test, y_train, y_test):
    """
    svm с нелинейным ядром
    :return:
    """
    clf_SVC = SVC(C=0.1, kernel='rbf', degree=3, gamma=1, coef0=0.0, shrinking=True,
                  probability=False, tol=0.001, cache_size=1000, class_weight=None,
                  verbose=True, max_iter=-1, decision_function_shape="ovr", random_state=0)
    clf_SVC.fit(X_train, y_train)

    print('Accuracy of SVC on training set: {:.2f}'.format(clf_SVC.score(X_train, y_train) * 100))
    print('Accuracy of SVC on test set: {:.2f}'.format(clf_SVC.score(X_test, y_test) * 100))

    # confusion_matrix
    print(confusion_matrix(clf_SVC.predict(X_test), y_test))


def build_model_rnn(embed_len):
    """
    построение модели rnn
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


def build_model_cnn(embed_len):
    """
    построение модели cnn
    :param embed_len: длина векторного представления
    :return:
    """
    model = Sequential()
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_model_rnn_with_embedding(max_features, embed_len):
    """
    построение модели rnn со слоем embedding
    :param max_features: кол во слов в алфавите
    :param embed_len: длина векторного представления
    :return:
    """
    model = models.Sequential()
    model.add(layers.Embedding(max_features, embed_len))
    model.add(layers.LSTM(embed_len))
    model.add(layers.Dense(1, activation='sigmoid'))
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

    if 'DESKTOP-TF87PFA' in os.environ['COMPUTERNAME']:
        glove_dir = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\glove.6B'
        imdb_dir: str = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\aclImdb'
        train_dir = os.path.join(imdb_dir, 'train')
        test_dir = os.path.join(imdb_dir, 'test')
        imdb_csv = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\csv_files\\imdb_mean.csv'
        to_imdb_csv = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\csv_files'

    else:
        glove_dir = 'D:\\datasets\\glove.6B'
        imdb_dir: str = 'D:\\datasets\\aclImdb'
        train_dir = os.path.join(imdb_dir, 'train')
        test_dir = os.path.join(imdb_dir, 'test')
        imdb_csv = 'D:\\datasets\\csv_files\\imdb_mean.csv'
        to_imdb_csv = 'D:\\datasets\\csv_files'
        rubcova_corpus_path = 'D:\datasets\\rubcova_corpus'
        rus_embeddings_path = 'D:\\datasets\\языковые модели\\180\\model.txt'

    stop_words = set(stopwords.words('russian'))

    # загрузка rus embeddings
    # embeddings_index = embeddings_download(rus_embeddings_path)

    # загрузка данных
    texts, labels = data_download(rubcova_corpus_path)

    # обучаем модель самостоятельно с помощью gensim
    texts = [tokenizer(text) for text in texts]
    mdl_gensim = Word2Vec(texts, min_count=0, size=300)

    # векторизация текстов
    X, y = text_vectorization(texts, labels, mdl_gensim)

    # разделение выборки на тренировочную и тестовую
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
    del X

    # nonlinear svm
    # nonlinear_svm(X_train, X_test, y_train, y_test)

    # RNN
    # mdl = build_model_rnn(300)
    #
    # history = mdl.fit(X_train,
    #                   y_train,
    #                   epochs=10,
    #                   batch_size=128,
    #                   validation_split=0.2)
    # loss_graph(history)
    # accuracy_graph(history)
    # print(mdl.evaluate(X_test, y_test))
    # print(mdl.summary())

    # CNN
    mdl = build_model_cnn(300)

    history = mdl.fit(X_train,
                      y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)
    loss_graph(history)
    accuracy_graph(history)
    print(mdl.evaluate(X_test, y_test))
    print(mdl.summary())

    total_time = round((time.time() - start_time))
    print("Time elapsed: %s minutes %s seconds" % ((total_time // 60), round(total_time % 60)))

    # generators
    # https: // realpython.com / introduction - to - python - generators /
    # https: // stanford.edu / ~shervine / blog / keras - how - to - generate - data - on - the - fly
    # https: // www.tutorialspoint.com / python / file_seek.htm