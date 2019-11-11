import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def tokenizer(text):
    """
    форматирование строки - замена пробелов, запятых, приведение к нижнему регистру, разделение на слова по пробелам
    """
    text = re.sub('<[^>]*>', '', text)
    text = re.sub(r'[a-z]([A-Z])', r'-\1', text.replace('.', ' .').replace(',', ' ,')).lower().split()
    return text


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

    for subdir, dirs, files in os.walk(newsgroup_dir):
        for dir in dirs:
            dict_labels[dir] = cnt
            cnt += 1

    for element in dict_labels:
        for subdir, dirs, files in os.walk(os.path.join(newsgroup_dir, element)):
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

    for i in range(len(texts)):
        X[i] = np.asarray(texts[i])

    y = np.array(labels)
    y = y[np.newaxis].T

    return np.append(X, y, axis=1)


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
    X_reduced = TSNE(n_components=2).fit_transform(X)
    sns.scatterplot(X_reduced[:, 0], X_reduced[:, 1], hue=y, legend='full')


glove_dir = 'D:\\datasets\\glove.6B'
newsgroup_dir: str = 'D:\\datasets\\20_newsgroups'

# загрузка embedding'ов
embeddings_index = embeddings_download(glove_dir)
# X = data_download_mean(embeddings_index, newsgroup_dir)
# pd.DataFrame(X).to_csv("20_newsgroups_mean.csv")

# загрузка данных, формирование тренировочной и тестовой выборок
newsgroups_data = np.genfromtxt('20_newsgroups_mean.csv', delimiter=',')
X = newsgroups_data[1:, 1:-1]
y = newsgroups_data[1:, -1:]

# масштабированimие выборок
scaler = StandardScaler()
scaler.fit_transform(X)

# разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

# построение трехмерного графика сниженной размерности
dim_reduction_plot_tsne(X_test, y_test)

# linear svm
# clf = LinearSVC(penalty='l2', loss='squared_hinge',
#                 dual=False, tol=0.0001, C=1, multi_class='ovr',
#                 fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0
#                 , random_state=0, max_iter=100000)
# clf.fit(X_train, y_train)
#
# print('Accuracy of linear SVC on training set: {:.2f}'.format(clf.score(X_train, y_train) * 100))
# print('Accuracy of linear SVC on test set: {:.2f}'.format(clf.score(X_test, y_test) * 100))

# nonlinear svm
clf_SVC = SVC(C=0.1, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True,
              probability=False, tol=0.001, cache_size=1000, class_weight=None,
              verbose=0, max_iter=-1, decision_function_shape="ovr", random_state=0)
clf_SVC.fit(X_train, y_train)

print('Accuracy of SVC on training set: {:.2f}'.format(clf_SVC.score(X_train, y_train) * 100))
clf_SVC.intercept_ = -140
print('Accuracy of SVC on test set: {:.2f}'.format(clf_SVC.score(X_test, y_test) * 100))

# optimal c parameter
# c = np.logspace(start=-15, stop=1000, base=1.02)
# param_grid = {'C': c}
#
# grid = GridSearchCV(clf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
# grid.fit(X_train, y_train)
#
# print("The best parameters are %s with a score of %0.0f" % (grid.best_params_, grid.best_score_ * 100))
# print("Best estimator accuracy on test set {:.2f} ".format(grid.best_estimator_.score(X_test, y_test) * 100))

# roc curve
# random_state = np.random.RandomState(0)
# classifier = OneVsRestClassifier(clf_SVC.SVC(kernel='linear', probability=True,
#                                              random_state=random_state))
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(2):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
# plt.figure()
# lw = 2
# plt.plot(fpr[2], tpr[2], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
