from rubcova_testing_models import *

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

    # загрузка данных
    texts, labels = data_download(rubcova_corpus_path)

    # разделение данных на тренировочную и тестовую выборки
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=2)

    # обучаем модель самостоятельно с помощью gensim
    own_model = Word2Vec(texts, min_count=0, size=300)

    embed_len = own_model.vector_size
    vocab_power = 100000
    sentence_len = 26

    # создаем и обучаем токенизатор
    tokenizer = Tokenizer(num_words=100000)
    tokenizer.fit_on_texts(X_train)

    # отображаем каждый текст в массив идентификаторов токенов
    x_train_seq = get_sequences(tokenizer, X_train)
    x_test_seq = get_sequences(tokenizer, X_test)

    # инициализируем матрицу embedding слоя нулями
    embedding_matrix = np.zeros((vocab_power, embed_len))

    # добавляем vocab_power=100000 наиболее часто встречающихся слов из обучающей выборки в embedding слой
    for word, i in tokenizer.word_index.items():
        if i >= vocab_power:
            break
        if word in own_model.wv.vocab.keys():
            embedding_matrix[i] = own_model.wv[word]

    # model building
    mdl = build_model_multi_cnn_with_embed(X_train.shape[1], X_train.shape[2], vocab_power, sentence_len,
                                           embedding_matrix)

    history = mdl.fit(X_train,
                      y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)
    loss_graph(history)
    accuracy_graph(history)
    print(mdl.evaluate(X_test, y_test))
    print(mdl.summary())

    # confusion matrix
    cm = confusion_matrix(y_test, np.around(mdl.predict(X_test)))
    plot_confusion_matrix(cm, cmap=plt.cm.Blues, my_tags=[0, 1])