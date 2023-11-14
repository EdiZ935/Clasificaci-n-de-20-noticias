import pickle
import spacy
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

nlp = spacy.load("en_core_web_sm")
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

x_train = newsgroups_train.data  # arreglo de datos de entrenamiento
y_train = newsgroups_train.target  # arreglo de etiquetas de entrenamiento
x_test = newsgroups_test.data  # arreglo de los documentos de prueba
y_test = newsgroups_test.target  # arreglo de etiquetas de prueba

df_train = pd.DataFrame({
    "doc": x_train,
    "label": y_train
})
df_test = pd.DataFrame({
    "doc": x_test,
    "label": y_test
})

with open("train.pkl", 'wb') as file:
    pickle.dump(df_train, file)
with open("test.pkl", 'wb') as file:
    pickle.dump(df_test, file)
