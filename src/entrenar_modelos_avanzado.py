# ==========================================
# MUNDIAL PREDICTOR ML
# PASO 11
# MODELOS CON DATASET AVANZADO
# ==========================================


import pandas as pd


from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression


from sklearn.ensemble import RandomForestClassifier


from xgboost import XGBClassifier


from sklearn.metrics import accuracy_score, classification_report


import joblib



print("\n==============================")
print(" MODELOS DATASET AVANZADO")
print("==============================\n")



# ==========================================
# CARGAR DATASET
# ==========================================


df = pd.read_csv(

"data/dataset_mundial_avanzado.csv"

)



print(df.head())



# ==========================================
# LIMPIEZA
# ==========================================


df = df.fillna(0)



# ==========================================
# VARIABLES
# ==========================================


X = df.drop(

"resultado",

axis=1

)



y = df["resultado"]



# ==========================================
# DIVISION DATOS
# ==========================================


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# ==========================================
# MODELOS
# ==========================================


modelos = {


"Regresion_Logistica":

LogisticRegression(

max_iter=3000

),



"Random_Forest":

RandomForestClassifier(

n_estimators=300,

random_state=42

),



"XGBoost":

XGBClassifier(

n_estimators=300,

learning_rate=0.05,

max_depth=4,

random_state=42,

eval_metric="mlogloss"

)

}




resultados=[]



# ==========================================
# ENTRENAMIENTO
# ==========================================


for nombre, modelo in modelos.items():


    print("\n====================")

    print(nombre)

    print("====================")


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



    print("\nPrecision:")

    print(precision)



    print("\nReporte:")

    print(

        classification_report(

            y_test,

            pred

        )

    )



    resultados.append(

        [

        nombre,

        precision

        ]

    )



    joblib.dump(

        modelo,

        f"models/{nombre}_avanzado.pkl"

    )



    print(

    "Guardado:",

    f"models/{nombre}_avanzado.pkl"

    )





# ==========================================
# COMPARACION
# ==========================================


tabla = pd.DataFrame(

resultados,

columns=[

"Modelo",

"Precision"

]

)



print("\n==============================")

print("RESULTADOS")

print("==============================")

print(tabla)



tabla.to_csv(

"data/resultados_avanzados.csv",

index=False

)



print("\nArchivo creado:")
print("data/resultados_avanzados.csv")
