import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#para rodar: python teste_svm.py no terminal

# listas
textos = []
labels = []

# caminho das pastas
fake_path = "Fake.br-Corpus/Fake.br-Corpus-master/full_texts/fake"
true_path = "Fake.br-Corpus/Fake.br-Corpus-master/full_texts/true"

# ler fake news
for arquivo in os.listdir(fake_path):

    with open(os.path.join(fake_path, arquivo), encoding="utf-8") as f:

        textos.append(f.read())
        labels.append(1)

# ler notícias reais
for arquivo in os.listdir(true_path):

    with open(os.path.join(true_path, arquivo), encoding="utf-8") as f:

        textos.append(f.read())
        labels.append(0)

# dataframe
df = pd.DataFrame({
    "texto": textos,
    "label": labels
})

print(df.head())

# transformar texto em números
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(df["texto"])
y = df["label"]

# dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# criar modelo SVM
modelo = SVC()

# treinar
modelo.fit(X_train, y_train)

# prever
predicoes = modelo.predict(X_test)

# resultados
print("\nACURÁCIA:")
print(accuracy_score(y_test, predicoes))

print("\nRELATÓRIO:")
print(classification_report(y_test, predicoes))