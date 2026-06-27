# ==========================================
# MUNDIAL PREDICTOR ML
# SIMULACION MONTE CARLO
# ==========================================


import pandas as pd

import joblib

import random



print("\n==============================")
print(" MONTE CARLO MUNDIAL ML")
print("==============================\n")



# cargar modelo

modelo = joblib.load(

"models/XGBoost_avanzado.pkl"

)



# cargar equipos

equipos = pd.read_csv(

"data/equipos_mundial.csv"

)



equipos_dict = equipos.set_index(

"equipo"

).to_dict(

"index"

)



# ==========================================
# PARTIDO
# ==========================================


def jugar(local, visitante):


    e1 = equipos_dict[local]

    e2 = equipos_dict[visitante]



    partido = pd.DataFrame([{

        "ranking_local": e1["ranking_fifa"],

        "ranking_visitante": e2["ranking_fifa"],

        "valor_local": e1["valor_plantilla"],

        "valor_visitante": e2["valor_plantilla"],

        "champions_local": e1["jugadores_champions"],

        "champions_visitante": e2["jugadores_champions"],

        "edad_local": e1["edad_promedio"],

        "edad_visitante": e2["edad_promedio"],

        "ataque_local": e1["ataque"],

        "ataque_visitante": e2["ataque"],

        "defensa_local": e1["defensa"],

        "defensa_visitante": e2["defensa"]

    }])



    pred = modelo.predict_proba(

        partido

    )[0]



    resultado = random.choices(

        [visitante,"empate",local],

        weights=[

            pred[0],

            pred[1],

            pred[2]

        ]

    )[0]



    if resultado=="empate":

        return random.choice(

            [local,visitante]

        )


    return resultado




# ==========================================
# TORNEO
# ==========================================


def mundial():



    equipos=[


    "Argentina",

    "Francia",

    "Brasil",

    "Inglaterra",

    "España",

    "Alemania",

    "Portugal",

    "Uruguay"


    ]



    # cuartos

    ronda1=[]


    for i in range(0,8,2):

        ronda1.append(

            jugar(

            equipos[i],

            equipos[i+1]

            )

        )



    # semifinal

    ronda2=[]


    for i in range(0,4,2):

        ronda2.append(

            jugar(

            ronda1[i],

            ronda1[i+1]

            )

        )



    # final

    campeon = jugar(

        ronda2[0],

        ronda2[1]

    )


    return campeon




# ==========================================
# 1000 MUNDIALES
# ==========================================


resultados=[]


simulaciones=1000



for i in range(simulaciones):


    campeon = mundial()


    resultados.append(

        campeon

    )



conteo = pd.Series(

resultados

).value_counts()



probabilidades = (

conteo / simulaciones * 100

)



print("\n==============================")

print(" PROBABILIDAD CAMPEON")

print("==============================\n")



print(

probabilidades.round(2)

)



probabilidades.to_csv(

"data/probabilidad_campeon.csv"

)



print("\nArchivo creado:")

print(

"data/probabilidad_campeon.csv"

)
