import pandas as pd


from sklearn.model_selection import cross_val_score


from sklearn.pipeline import Pipeline


from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import LogisticRegression


from sklearn.ensemble import RandomForestClassifier


from xgboost import XGBClassifier



print("==============================")
print(" MODELOS MEJORADOS")
print("==============================")



df = pd.read_csv(

"data/dataset_mundial_avanzado.csv"

)



X = df.drop(

"resultado",

axis=1

)


y = df["resultado"]



modelos = {


"Logistica":

Pipeline([

("scaler",StandardScaler()),

("modelo",

LogisticRegression(

max_iter=5000,

class_weight="balanced"

))

]),



"RandomForest":

RandomForestClassifier(

n_estimators=500,

class_weight="balanced",

random_state=42

),



"XGBoost":

XGBClassifier(

n_estimators=500,

learning_rate=0.03,

max_depth=3,

eval_metric="mlogloss"

)

}



for nombre, modelo in modelos.items():


    scores = cross_val_score(

        modelo,

        X,

        y,

        cv=5,

        scoring="accuracy"

    )


    print("\n",nombre)

    print("Resultados:")

    print(scores)

    print("Promedio:")

    print(scores.mean())
