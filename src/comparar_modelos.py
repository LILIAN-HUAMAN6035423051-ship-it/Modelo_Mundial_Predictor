# ==========================================
# MUNDIAL PREDICTOR ML
# PASO 9: COMPARACION DE MODELOS
# ==========================================


import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import matplotlib.pyplot as plt



print("\n==============================")
print(" COMPARACION DE MODELOS ML")
print("==============================\n")



# ==========================================
# CARGAR DATOS
# ==========================================


df = pd.read_csv(
    "data/dataset_ml.csv"
)



# Variables

X = df[

[
"goles_local",
"goles_visitante",
"diferencia_goles"
]

]


y = df["resultado"]



# Separar datos


X_train, X_test, y_train, y_test = train_test_split(

X,

y,

test_size=0.2,

random_state=42

)



# ==========================================
# CREAR MODELOS
# ==========================================


modelos = {


"Regresion Logistica":

LogisticRegression(
max_iter=1000
),


"Random Forest":

RandomForestClassifier(
n_estimators=200,
random_state=42
),


"XGBoost":

XGBClassifier(

n_estimators=200,

learning_rate=0.05,

max_depth=5,

random_state=42,

eval_metric="mlogloss"

)

}



# ==========================================
# ENTRENAR Y COMPARAR
# ==========================================


resultados = []



for nombre, modelo in modelos.items():


    print("\nEntrenando:", nombre)


    modelo.fit(

    X_train,

    y_train

    )


    pred = modelo.predict(

    X_test

    )


    precision = accuracy_score(

    y_test,

    pred

    )


    resultados.append(

    [

    nombre,

    precision

    ]

    )



# Crear tabla

tabla = pd.DataFrame(

resultados,

columns=[

"Modelo",

"Precision"

]

)



print("\n==============================")

print(tabla)


print("==============================")



# ==========================================
# GRAFICO
# ==========================================


plt.figure(figsize=(8,5))


plt.bar(

tabla["Modelo"],

tabla["Precision"]

)



plt.ylabel(

"Precision"

)


plt.xlabel(

"Modelo"

)


plt.title(

"Comparacion de modelos Mundial Predictor"

)


plt.ylim(

0,

1

)


plt.xticks(

rotation=20

)


plt.show()



# Guardar resultados


tabla.to_csv(

"data/comparacion_modelos.csv",

index=False

)


print("\nArchivo creado:")

print(

"data/comparacion_modelos.csv"

)
