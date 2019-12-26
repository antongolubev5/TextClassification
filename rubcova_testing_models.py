import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from keras import layers, models, Sequential
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from keras.models import load_model


def tweet_tokenizer(text, use_stop_words, stemming):
    """
    форматирование строки:
    приведение к нижнему регистру;
    удаление интернет-ссылок и упоминания имен пользователей через @
    удаление знаков пунктуации;
    удаление стоп слов;
    удаление дат и знаков пунктуации
    оставляем смайлики
    токенизация (встроенный nltk.tweet tokenizer)
    """

    tw_tok = TweetTokenizer()
    result = ' '.join(tw_tok.tokenize(text))

    result = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', result)
    result = re.sub('@[^\s]+', 'USER', result)
    result = re.sub('[^a-zA-Zа-яА-Я:):(]+', ' ', result)

    result = result.split(" ")

    # punc_list = string.punctuation + '0123456789'
    # t = str.maketrans(dict.fromkeys(punc_list, " "))
    # text = text.lower().translate(t)

    stemmer = SnowballStemmer("russian")

    if use_stop_words and stemming:
        output = [stemmer.stem(element) for element in result if not (element in stop_words)]

    if use_stop_words:
        output = [element.lower() for element in result if not (element in stop_words)]

    if stemming:
        output = [stemmer.stem(element) for element in result]

    return output


def embeddings_download(file_path):
    """
    загрузка представлений слов русскоязычной модели с rusvectores
    :param file_path: директория с файлами
    :return: model
    """
    model = KeyedVectors.load_word2vec_format(file_path, binary=True)
    # embeddings_index = {}
    # f = open(file_path, 'r', encoding='utf-8')
    #
    # for line in f:
    #     values = line.split()
    #     word = values[0].split("_")[0]
    #     coeffs = np.asarray(values[1:], dtype='float32')
    #     embeddings_index[word] = coeffs
    # f.close()

    return model


def data_download(file_path):
    """
    загрузка текстов и меток
    :param file_path: директория с двумя csv-файлами
    :return: список raw текстов и список меток
    """
    data_positive = pd.read_csv(file_path + '\\positive.csv', sep=';', encoding='utf-8', header=None)
    data_negative = pd.read_csv(file_path + '\\negative.csv', sep=';', encoding='utf-8', header=None)
    corpus = pd.concat((data_positive, data_negative), axis=0)
    corpus = corpus.dropna()
    corpus.columns = ["id", "tda", "tname", "ttext", "ttype", "trep", "trtw", "tfav", "tstcount", "tfoll", "tfrien",
                      "listcount"]
    corpus.drop(
        columns=["id", "tda", "tname", "ttype", "trep", "trtw", "tfav", "tstcount", "tfoll", "tfrien", "listcount"],
        axis=1)

    corpus['ttext_preproc'] = corpus.ttext.apply(lambda x: tweet_tokenizer(x, use_stop_words=True, stemming=False))
    # corpus['ttext_preproc'] = corpus.ttext_preproc.apply(lambda x: ' '.join(x))

    # corpus.to_csv(file_path + '\\preprocessed.csv', columns=["ttext" "ttext_preproc"])

    texts = list(corpus.ttext_preproc)
    labels = [1] * len(data_positive) + [0] * len(data_negative)

    return texts, labels


def text_vectorization(texts, labels, embeddings):
    """
    векторизация текстов с помощью предварительно обученных embeddings
    :param embeddings: предварительно обученные embeddings
    :param labels: метки текстов
    :param texts: тексты
    :return:
    """

    # tokenization
    # texts = [tweet_tokenizer(text, use_stop_words=True, stemming=False) for text in texts]

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

    embed_len = len(embeddings['лес'])
    text_len = 12

    y = np.asarray(labels)
    X = np.zeros((len(texts), text_len, embed_len), dtype=np.float16)

    for i in range(len(texts)):
        for j in range(min(len(texts[i]), text_len)):
            # if texts[i][j] in embeddings.index2word:
            X[i][j] = embeddings[texts[i][j]]

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


def build_model_cnn():
    """
    построение модели cnn
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


def plot_confusion_matrix(cm, cmap=plt.cm.Blues, my_tags=[0, 1]):
    """
    построение матрицы ошибок в форме оттенков цветов
    :param cm: вычисленная матрица ошибок
    :param cmap: карта цветов
    :param my_tags: метки классов
    :return:
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(my_tags))
    target_names = my_tags
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def predict_tweet(tweet, predict_model, text_model):
    """
    определение тональности твита на обученной сети
    :param text_model: обученная модель для представления текстов
    :param predict_model: обученная модель
    :param tweet: твит
    :return: positive/negative
    """
    # preprocessing
    tweet = tweet_tokenizer(tweet, use_stop_words=True, stemming=False)

    # vectorization
    text_len = 12
    embed_len = len(own_model.wv['лес'])

    X = np.zeros((1, text_len, embed_len), dtype=np.float16)

    for i in range(min(len(tweet), text_len)):
        X[0][i] = text_model.wv[tweet[i]]

    # prediction
    if np.round(predict_model.predict(X)) == 0:
        answer = 'negative'
    else:
        answer = 'positive'

    return answer


if __name__ == "__main__":

    start_time = time.time()

    if 'DESKTOP-TF87PFA' in os.environ['COMPUTERNAME']:
        rubcova_corpus_path = 'D:\datasets\\rubcova_corpus'
        rus_embeddings_path = 'D:\\datasets\\языковые модели\\'
        save_arrays_path = 'D:\\datasets\\npy\\rubcova_corpus\\'
        save_model_path = 'D:\\datasets\\models\\'

    else:
        rubcova_corpus_path = 'D:\\datasets\\rubcova_corpus'
        rus_embeddings_path = 'D:\\datasets\\языковые модели\\'
        save_arrays_path = 'D:\\datasets\\npy\\rubcova_corpus\\'
        save_model_path = 'D:\\datasets\\models\\'

    stop_words = set(stopwords.words('russian'))

    # загрузка rus embeddings
    # embeddings_index = embeddings_download(rus_embeddings_path + '180\\model.bin')

    # загрузка данных
    # texts, labels = data_download(rubcova_corpus_path)

    # обучаем модель самостоятельно с помощью gensim
    # own_model = Word2Vec(texts, min_count=0, size=300)
    # own_model.save(rus_embeddings_path + 'tweets.model')

    # загрузка обученной текстовой модели
    own_model = Word2Vec.load(rus_embeddings_path + 'tweets.model')

    # векторизация текстов
    # X, y = text_vectorization(texts, labels, own_model)

    # сохранение векторизованных текстов для дальнейшего использования
    # np.save(save_arrays_path + '\\X.npy', X)
    # np.save(save_arrays_path + '\\y.npy', y)

    # загрузка векторизованных текстов
    X = np.load(save_arrays_path + '\\X.npy')
    y = np.load(save_arrays_path + '\\y.npy')

    # разделение выборки на  тренировочную и тестовую
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
    del X

    # nonlinear svm
    # nonlinear_svm(X_train, X_test, y_train, y_test)

    # RNN
    # mdl = build_model_rnn(300)
    #
    # history = mdl.fit(X_train,
    #                   y_train,
    #                   epochs=8,
    #                   batch_size=128,
    #                   validation_split=0.2)
    # loss_graph(history)
    # accuracy_graph(history)
    # print(mdl.evaluate(X_test, y_test))
    # print(mdl.summary())

    # CNN
    # mdl = build_model_cnn()
    #
    # history = mdl.fit(X_train,
    #                   y_train,
    #                   epochs=15,
    #                   batch_size=128,
    #                   validation_split=0.2)
    # loss_graph(history)
    # accuracy_graph(history)
    # print(mdl.evaluate(X_test, y_test))
    # print(mdl.summary())

    # mdl.save(save_model_path + 'cnn.h5')

    # загрузка обученной предсказывающей модели
    mdl = load_model(save_model_path + 'cnn.h5')

    # проверка классификации
    # tweet = "какой хороший сегодня день"
    # print(tweet, predict_tweet(tweet, mdl, own_model))

    # confusion matrix
    cm = confusion_matrix(y_test, np.around(mdl.predict(X_test)))
    plot_confusion_matrix(cm, cmap=plt.cm.Blues, my_tags=[0, 1])

    total_time = round((time.time() - start_time))
    print("Time elapsed: %s minutes %s seconds" % ((total_time // 60), round(total_time % 60)))
