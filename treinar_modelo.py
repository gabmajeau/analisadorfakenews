import os
import pandas as pd
import joblib
from sklearn.svm import SVC
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from unidecode import unidecode


nltk.download('stopwords')

stop_words = stopwords.words('portuguese')

def limpar_texto(texto):

    texto = texto.lower()

    texto = unidecode(texto)

    texto = re.sub(r"http\S+", "", texto)

    texto = re.sub(r"[^a-zA-Z\s]", "", texto)

    palavras = texto.split()

    palavras = [
        palavra
        for palavra in palavras
        if palavra not in stop_words
    ]

    return " ".join(palavras)

# listas
textos = []
labels = []

# caminhos
fake_path = "Fake.br-Corpus/Fake.br-Corpus-master/full_texts/fake"
true_path = "Fake.br-Corpus/Fake.br-Corpus-master/full_texts/true"

# fake news
for arquivo in os.listdir(fake_path):

    with open(os.path.join(fake_path, arquivo), encoding="utf-8") as f:

        texto_limpo = limpar_texto(f.read())

        textos.append(texto_limpo)
        labels.append(1)

# notícias reais
for arquivo in os.listdir(true_path):

    with open(os.path.join(true_path, arquivo), encoding="utf-8") as f:

        texto_limpo = limpar_texto(f.read())

        textos.append(texto_limpo)
        labels.append(0)

# dataframe
df = pd.DataFrame({
    "texto": textos,
    "label": labels
})

# vetorizar
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["texto"])
y = df["label"]

# dividir
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# modelo SVM
modelo = SVC(
    probability=True,
    kernel='linear',
    class_weight='balanced'
)

# treinar
modelo.fit(X_train, y_train)

# salvar modelo
joblib.dump(modelo, "modelo_fake_news.pkl")

# salvar vetorizer
joblib.dump(vectorizer, "vectorizer.pkl")

print("MODELO TREINADO E SALVO!")