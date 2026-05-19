from flask import Flask, render_template, request
import joblib
import numpy as np

# carregar modelo
modelo = joblib.load("modelo_fake_news.pkl")

# carregar vetorizador
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():

    resultado = None
    porcentagem = None
    nivel = None
    texto_destacado = None
    motivos = []

    if request.method == "POST":

        texto_original = request.form["noticia"]

        # vetorizar
        texto_vetorizado = vectorizer.transform([texto_original])

        # previsão
        predicao = modelo.predict(texto_vetorizado)

        # distância do SVM
        distancia = modelo.decision_function(texto_vetorizado)[0]

        prob_fake = min(max((distancia + 1) * 50, 0), 100)

        porcentagem = round(prob_fake, 2)

        # classificação
        if prob_fake >= 50:
            resultado = "Possível Fake News"
        else:
            resultado = "Possível Notícia Verdadeira"

        # nível
        if porcentagem < 40:
            nivel = "ALTO"

        elif porcentagem < 70:
            nivel = "MÉDIO"

        else:
            nivel = "BAIXO"

        # destacar palavras
        texto_destacado = texto_original

        try:

            feature_names = vectorizer.get_feature_names_out()

            coeficientes = modelo.coef_[0]

            indices_importantes = np.argsort(coeficientes)[-20:]

            palavras_importantes = [
                feature_names[i]
                for i in indices_importantes
            ]

            for palavra in palavras_importantes:

                texto_destacado = texto_destacado.replace(
                    palavra,
                    f"<span class='highlight'>{palavra}</span>"
                )

        except:
            pass

        # motivos da classificação

        texto_lower = texto_original.lower()

        palavras_sensacionalistas = [
            "urgente",
            "chocante",
            "segredo",
            "revelado",
            "escândalo",
            "absurdo",
            "inacreditável",
            "alerta"
        ]

        contador = 0

        for palavra in palavras_sensacionalistas:

            if palavra in texto_lower:
                contador += 1

        # fake
        if prob_fake >= 50:

            if contador > 0:
                motivos.append(
                    "Uso de linguagem sensacionalista."
                )

            if "!" in texto_original:
                motivos.append(
                    "Uso de pontuação emocional."
                )

            if len(texto_original.split()) < 80:
                motivos.append(
                    "Texto curto e com poucas informações verificáveis."
                )

            motivos.append(
                "Estrutura textual semelhante às fake news do treinamento."
            )

        # real
        else:

            motivos.append(
                "Estrutura semelhante a notícias jornalísticas."
            )

            motivos.append(
                "Linguagem mais objetiva e informativa."
            )

            if contador == 0:
                motivos.append(
                    "Baixo uso de termos sensacionalistas."
                )

    return render_template(

        "index.html",

        resultado=resultado,
        porcentagem=porcentagem,
        nivel=nivel,
        texto_destacado=texto_destacado,
        motivos=motivos
    )

if __name__ == "__main__":
    app.run(debug=True)