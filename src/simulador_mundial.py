# ==========================================
# MUNDIAL PREDICTOR ML
# SIMULADOR CORREGIDO
# ==========================================


import pandas as pd

import joblib

import random



print("\n==============================")
print(" SIMULADOR MUNDIAL ML")
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
# FUNCION PARTIDO
# ==========================================


def jugar_partido(local, visitante):


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



    resultado = modelo.predict(

        partido

    )[0]



    if resultado == 2:

        ganador = local


    elif resultado == 0:

        ganador = visitante


    else:

        ganador = random.choice(

            [local, visitante]

        )



    return ganador




# ==========================================
# RONDA
# ==========================================


def simular_ronda(equipos):


    ganadores=[]



    for i in range(0,len(equipos),2):


        e1 = equipos[i]

        e2 = equipos[i+1]



        ganador = jugar_partido(

            e1,

            e2

        )



        print(

            e1,

            "vs",

            e2,

            "->",

            ganador

        )



        ganadores.append(

            ganador

        )



    return ganadores




# ==========================================
# TORNEO
# ==========================================


equipos_mundial=[


"Argentina",

"Francia",

"Brasil",

"Inglaterra",

"España",

"Alemania",

"Portugal",

"Uruguay"

]



print("\nCUARTOS")

cuartos = simular_ronda(

equipos_mundial

)



print("\nSEMIFINAL")

semifinal = simular_ronda(

cuartos

)



print("\nFINAL")

finalista1 = semifinal[0]

finalista2 = semifinal[1]



campeon = jugar_partido(

finalista1,

finalista2

)



print(

finalista1,

"vs",

finalista2,

"->",

campeon

)



print("\n==============================")

print(" CAMPEON DEL MUNDIAL")

print("==============================")

print(campeon)
