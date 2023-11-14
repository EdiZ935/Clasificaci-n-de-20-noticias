import spacy
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
import numpy as np


nlp = spacy.load("en_core_web_sm")


def lematizar_documento(ruta_doc, nombre_archivo_salida):
    try:
        archivo_df = open(ruta_doc, 'rb')
        df = pickle.load(archivo_df)

        for i in range(len(df)):
            tokens = nlp(df.loc[i, 'doc'])
            lista_lemmas = []
            for t in tokens:
                lista_lemmas.append(t.lemma_)
            df.loc[i, 'doc'] = " ".join(lista_lemmas)

        salida = open(nombre_archivo_salida, 'wb')
        pickle.dump(df, salida)

        archivo_df.close()
        salida.close()
    except FileNotFoundError:
        print("No se encontro el archivo")


def quitar_sw_documento(ruta_doc, nombre_archivo_salida):
    try:
        archivo_df = open(ruta_doc, 'rb')
        df = pickle.load(archivo_df)

        for i in range(len(df)):
            tokens = nlp(df.loc[i, 'doc'])
            lista_palabras = []
            for t in tokens:
                if not t.is_stop:  # and t.pos_ != "DET" and t.pos_ != "ADP" and t.pos_ != "CONJ" and t.pos_ != "PRON":
                    lista_palabras.append(t.text)
            texto = " ".join(lista_palabras)
            texto = re.sub('\s+', ' ', texto)
            texto = texto.strip()
            df.loc[i, 'doc'] = texto

        salida = open(nombre_archivo_salida, 'wb')
        pickle.dump(df, salida)
        print(df.loc[0, 'doc'])
        archivo_df.close()
        salida.close()
    except FileNotFoundError:
        print("No se encontro el archivo")


def limpia_texto(ruta_doc, nombre_archivo_salida):
    try:
        archivo_df = open(ruta_doc, 'rb')
        df = pickle.load(archivo_df)

        for i in range(len(df)):
            tokens = nlp(df.loc[i, 'doc'])
            lista_palabras = []
            for t in tokens:
                if not t.is_punct:
                    lista_palabras.append(t.text)
            texto = " ".join(lista_palabras)
            texto = re.sub('\s+', ' ', texto)
            texto = texto.strip()
            df.loc[i, 'doc'] = texto

        salida = open(nombre_archivo_salida, 'wb')
        pickle.dump(df, salida)
        print(df.loc[0, 'doc'])
        archivo_df.close()
        salida.close()
    except FileNotFoundError:
        print("No se encontro el archivo")


# Normalizacion combinaciones :
# quitar_sw_documento("test.pkl", "test_nsw.pkl")
# quitar_sw_documento("train.pkl", "train_nsw.pkl")

# lematizar_documento("test.pkl", "test_lemma.pkl")
# lematizar_documento("train.pkl", "train_lemma.pkl")

# limpia_texto('test.pkl','test_limp.pkl')
# limpia_texto('train.pkl','train_limp.pkl')

# lematizar_documento("test_nsw.pkl","test_nsw_lemma.pkl")
# lematizar_documento("train_nsw.pkl","train_nsw_lemma.pkl")

# quitar_sw_documento("test_lemma.pkl","test_lemma_nsw.pkl")
# quitar_sw_documento("train_lemma.pkl","train_lemma_nsw.pkl")

# limpia_texto('test_nsw.pkl','test_nsw_limp.pkl')
# limpia_texto('train_nsw.pkl','train_nsw_limp.pkl')

# lematizar_documento("test_limp.pkl","test_limp_lemma.pkl")
# lematizar_documento("train_limp.pkl","train_limp_lemma.pkl")

# lematizar_documento("test_nsw_limp.pkl","test_nsw_limp_lemma.pkl")
# lematizar_documento("train_nsw_limp.pkl","train_nsw_limp_lemma.pkl")


def get_x_y_data(ruta_doc):
    try:
        archivo_df = open(ruta_doc, 'rb')
        df = pickle.load(archivo_df)
        data = df['doc']
        target = df['label']
        archivo_df.close()

        return data, target
    except FileNotFoundError:
        print("No se encontro el archivo")


def rep_vec(data_train, data_test, vectorizer):
    vectors_train = vectorizer.fit_transform(data_train)
    vectors_test = vectorizer.transform(data_test)
    print(vectorizer.get_feature_names_out())
    print(len(vectorizer.get_feature_names_out()))
    return vectors_train, vectors_test


def get_model(route_train):
    X_train, y_train = get_x_y_data(route_train)
    sentences = [sentence.lower().split() for sentence in X_train]
    model = Word2Vec(sentences, vector_size = 300, window = 16, min_count = 5, workers = 4)
    # workers = number of threads

    return model


def vectorize_embedding(sentence, w2v_model):
    words = sentence.split()
    words_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(words_vecs) == 0:
        return np.zeros(300)
    words_vecs = np.array(words_vecs)
    return words_vecs.mean(axis=0)


def classify(clf, route_train, route_test, vectorizer=None):
    X_train, y_train = get_x_y_data(route_train)
    X_test, y_test = get_x_y_data(route_test)

    if vectorizer != None:
        vectors_train, vectors_test = rep_vec(X_train, X_test, vectorizer)
        clf.fit(vectors_train, y_train)
        y_pred = clf.predict(vectors_test)
    else:
        model = get_model(route_train)
        X_train = np.array([vectorize_embedding(sentence.lower(), model) for sentence in X_train])
        X_test = np.array([vectorize_embedding(sentence.lower(), model) for sentence in X_test])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    print(y_pred)
    print(classification_report(y_test, y_pred))


def classify_svd(clf, route_train, route_test, vectorizer):
    X_train, y_train = get_x_y_data(route_train)
    X_test, y_test = get_x_y_data(route_test)
    pipe = Pipeline([('text_representation', vectorizer), ('dimensionality_reduction', TruncatedSVD(300)),
                     ('classifier', clf)])
    pipe.set_params(dimensionality_reduction__n_components = 1000)
    print(pipe)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))


vectorizer_bin = CountVectorizer(binary=True)
vectorizer_freq = CountVectorizer()
vectorizer_tfidf = TfidfVectorizer()

clf_NB = MultinomialNB()
clf_LR = LogisticRegression(max_iter = 300)
clf_MLPC = MLPClassifier()
clf_SGD = SGDClassifier()

# Pruebas con los clasificadores:

print()
print()
print("No Stop Words + lemmatization : tfidf , SGD")
classify(clf_SGD, 'train_nsw_lemma.pkl', 'test_nsw_lemma.pkl', vectorizer_tfidf)
print()
print("No Stop Words + clean text + lemmatization : tfidf , SGD")
classify(clf_SGD, 'train_nsw_limp_lemma.pkl', 'test_nsw_limp_lemma.pkl', vectorizer_tfidf)

print()
print()
print("No Stop Words + lemmatization : tfidf , SGD , SVD")
classify_svd(clf_SGD, 'train_nsw_lemma.pkl', 'test_nsw_lemma.pkl', vectorizer_tfidf)
print()
print("No Stop Words + clean text + lemmatization : tfidf , SGD , SVD")
classify_svd(clf_SGD, 'train_nsw_limp_lemma.pkl', 'test_nsw_limp_lemma.pkl', vectorizer_tfidf)
