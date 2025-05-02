import pandas as pd
from transformers import pipeline

classifier = pipeline("text-classification", model="Hate-speech-CNERG/dehatebert-mono-english")

def clasificar_texto(texto):
    resultado = classifier(texto)
    etiqueta = resultado[0]['label']
    #score = resultado[0]['score']
    
    if etiqueta == 'HATE':
        return f"discurso de odio"
    elif etiqueta == 'OFF':
        return f"Contenido inapropiado"
    else:
        return f"Inofensivo"


def procesar_csv(ruta_csv):
    df = pd.read_csv(ruta_csv)
    df['etiqueta'] = df['texto'].apply(clasificar_texto)
    return df

csv1 = "resultados_train.csv"
csv2 = "resultados_validation.csv"
df1 = procesar_csv(csv1)
df2 = procesar_csv(csv2)
df1.to_csv("train.csv", index=False)
df2.to_csv("validation.csv", index=False)
