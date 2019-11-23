import os
import time
#d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tok import word_tokenize


def tokenizer(text):
    """
    ôîðìàòèðîâàíèå ñòðîêè https://github.com/kootenpv/tok/blob/master/README.md
    regexp, stop_words, lowercase, stemmer
    """
    ps = PorterStemmer()
    result = word_tokenize(text)
    drop = []
    for element in result:
        if not (element in stop_words):
            drop.append(ps.stem(element).lower())

    return drop


def tokenizer_tfidf(text):
    """
    ôîðìàòèðîâàíèå ñòðîêè https://github.com/kootenpv/tok/blob/master/README.md
    """
    # regexp, stop_words, lowercase, stemmer
    f = open('english_stop_words', 'r')
    stop_words = [line.strip() for line in f]

    ps = PorterStemmer()
    result = word_tokenize(text)
    drop = []
    for element in result:
        if not (element in stop_words):
            drop.append(ps.stem(element).lower())

    return ' '.join(drop)


def csv_from_txts(directory):
    """
    Çàãðóçêà äàííûõ â csv file.
    Áåðåì ñðåäíèé âåêòîð âñåãî äîêóìåíòà äëÿ ïîëó÷åíèÿ áîëåå íèçêîé ðàçìåðíîñòè
    :param directory: äèðåêòîðèÿ ñ ôàéëàìè
    :return: îáó÷àþùàÿ è òðåíèðîâî÷íàÿ âûáîðêè
    """
    df = pd.DataFrame(columns=['text', 'label'])
    num = 0
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(directory, label_type)
        for file_name in os.listdir(dir_name):
            if file_name[-4:] == '.txt':
                f = open(os.path.join(dir_name, file_name), encoding='utf-8')
                df.loc[num] = [f.read(), 0 if label_type == 'neg' else 1]
                f.close()
                num += 1

    return df


def dim_reduction_plot(X, y):
    """
    ïîñòðîåíèå òðåõìåðíîãî ãðàôèêà ñíèæåííîé ðàçìåðíîñòè
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
    Ñòîõàñòè÷åñêîå âëîæåíèå ñîñåäåé ñ t-ðàñïðåäåëåíèåì (àíãë. t-distributed Stochastic Neighbor Embedding, t-SNE) —
    ýòî àëãîðèòì ìàøèííîãî îáó÷åíèÿ äëÿ âèçóàëèçàöèè, ðàçðàáîòàííûé Ëîðåíñîì âàí äåð Ìààòåíîì è Äæåôôðè Õèíòîíîì.
    Îí ÿâëÿåòñÿ òåõíèêîé íåëèíåéíîãî ñíèæåíèÿ ðàçìåðíîñòè, õîðîøî ïîäõîäÿùåé äëÿ âëîæåíèÿ äàííûõ âûñîêîé ðàçìåðíîñòè
    äëÿ âèçóàëèçàöèè â ïðîñòðàíñòâî íèçêîé ðàçìåðíîñòè (äâóõ- èëè òðåõìåðíîå). Â ÷àñòíîñòè, ìåòîä ìîäåëèðóåò êàæäûé
    îáúåêò âûñîêîé ðàçìåðíîñòè äâóõ- èëè òð¸õìåðíîé òî÷êîé òàêèì îáðàçîì, ÷òî ïîõîæèå îáúåêòû ìîäåëèðóþòñÿ áëèçêî
    ðàñïîëîæåííûìè òî÷êàìè, à íåïîõîæèå òî÷êè ìîäåëèðóþòñÿ ñ áîëüøîé P òî÷êàìè, äàëåêî äðóã îò äðóãà îòñòîÿùèìè.
    """
    x_reduced = TSNE(n_components=2).fit_transform(X)
    sns.scatterplot(x_reduced[:, 0], x_reduced[:, 1], hue=y, legend='full')


def roc_curve_own(model, X, y):
    """
    ïîñòðîåíèå roc-êðèâîé ìîäåëè, èçìåíÿåìûé ïàðàìåòð - w_0
    :param model: îáó÷åííàÿ ìîäåëü
    :param X: ïðèçíàêè
    :param y: îòâåòû
    :return: ãðàôèê roc-êðèâîé
    """
    roc_array = np.zeros((len(X), 2))  # ñâåðõó fpr, ñíèçó tpr
    a_y = np.zeros((len(X), 2))  # ñâåðõó decision_function(x_i), ñíèçó ìåòêà êëàññà
    m = np.zeros(len(X))

    # ñîðòèðîâêà äâóìåðíîãî ìàññèâà ïî ñòðîêå m
    m = model.decision_function(X)
    a_y[:, 0] = np.array(m.T).astype(float)  # np.ravel(m.T)
    a_y[:, 1] = y.T
    l_a_y = list(a_y)
    l_a_y = sorted(l_a_y, key=lambda x: x[0])
    a_y_sorted = np.asarray(l_a_y)

    # çàïîëíÿåì roc_array tpr è fpr
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


def grid_search_best_params():
    """
    ïîèñê îïòèìàëüíûõ ïàðàìåòðîâ svc ìîäåëè
    :return:
    """
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    grid.fit(X_train, y_train)

    print(grid.best_params_)
    print(grid.best_estimator_)

    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))


def tf_idf_representation(csv_file):
    texts = list(csv_file['0'])
    labels = list(csv_file['1'])

    # corpus = list(map(tokenizer_tfidf, texts))
    corpus = list(texts)
    vectorizer = TfidfVectorizer(ngram_range=[1, 2], decode_error='ignore', lowercase=True, stop_words='english',
                                 min_df=0.02)
    # vectorizer = TfidfVectorizer(stop_words="english", decode_error='ignore', lowercase=True, ngram_range=(1, 2))
    representations = vectorizer.fit_transform(corpus).toarray()

    return representations, labels


start_time = time.time()

if('DESKTOP-TF87PFA' in os.environ['COMPUTERNAME'] ):
    glove_dir = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\glove.6B'
    imdb_dir: str = 'C:\\Users\\Alexandr\\Documents\\NLP\\diplom\\datasets\\aclImdb'
    train_dir = os.path.join(imdb_dir, 'train')
    test_dir = os.path.join(imdb_dir, 'test')
else:
    imdb_dir: str = 'D:\\datasets\\aclImdb'
    train_dir = os.path.join(imdb_dir, 'train')
    test_dir = os.path.join(imdb_dir, 'test')
# region make_csv
# Xy_train = csv_from_txts(train_dir)
# Xy_test = csv_from_txts(test_dir)
# pd.DataFrame(np.append(Xy_train, Xy_test, axis=0)).to_csv("imdb.csv")
# endregion

# çàãðóçêà äàííûõ, ôîðìèðîâàíèå òðåíèðîâî÷íîé è òåñòîâîé âûáîðîê
imdb_data = pd.read_csv('imdb.csv')
X, y = tf_idf_representation(imdb_data)

# ìàñøòàáèðîâàíèå âûáîðîê
scaler = StandardScaler().fit_transform(X)

# ðàçäåëåíèå âûáîðêè íà òðåíèðîâî÷íóþ è òåñòîâóþ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

# grid_search_best_params()

# region äâóìåðíûé ãðàôèê tsne
# ïîñòðîåíèå ãðàôèêà ñíèæåííîé ðàçìåðíîñòè
# a = np.append(X_test[:50], X_test[-50:], axis=0)
# b = np.append(y_test[:50], y_test[-50:], axis=0)
# dim_reduction_plot_tsne(X_test, y_test)
# endregion

# nonlinear svm
# clf_SVC = SVC(C=0.1, kernel='rbf', degree=3, gamma=1, coef0=0.0, shrinking=True,
#                     probability=False, tol=0.001, cache_size=1000, class_weight=None,
#                     verbose=True, max_iter=-1, decision_function_shape="ovr", random_state=0)
clf_SVC = SVC()
clf_SVC.fit(X_train, y_train)

print('Accuracy of SVC on training set: {:.2f}'.format(clf_SVC.score(X_train, y_train) * 100))
print('Accuracy of SVC on test set: {:.2f}'.format(clf_SVC.score(X_test, y_test) * 100))

# roc_curve
# roc_curve_own(clf_SVC, X_test, y_test)

# confusion_matrix
conf_matrix = confusion_matrix(clf_SVC.predict(X_test), y_test)

total_time = round((time.time() - start_time))
print("Time elapsed: %s minutes %s seconds" % ((total_time // 60), round(total_time % 60)))
