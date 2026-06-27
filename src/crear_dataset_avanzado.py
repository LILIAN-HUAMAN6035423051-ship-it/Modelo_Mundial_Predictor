# ==========================================
# MUNDIAL PREDICTOR ML
# DATASET AVANZADO
# ==========================================


import pandas as pd



print("\n==============================")
print(" CREANDO DATASET AVANZADO")
print("==============================\n")



# cargar partidos

partidos = pd.read_csv(
    "data/results.csv"
)



# cargar equipos

equipos = pd.read_csv(
    "data/equipos_mundial.csv"
)



# diccionario de equipos

equipos_dict = equipos.set_index(
    "equipo"
).to_dict(
    "index"
)



datos=[]



for _, partido in partidos.iterrows():


    local = partido["home_team"]

    visitante = partido["away_team"]



    # verificar si existen

    if local in equipos_dict and visitante in equipos_dict:


        e_local = equipos_dict[local]

        e_visitante = equipos_dict[visitante]



        # resultado

        if partido["home_score"] > partido["away_score"]:

            resultado = 2


        elif partido["home_score"] < partido["away_score"]:

            resultado = 0


        else:

            resultado = 1



        datos.append({


        "ranking_local":
        e_local["ranking_fifa"],


        "ranking_visitante":
        e_visitante["ranking_fifa"],


        "valor_local":
        e_local["valor_plantilla"],


        "valor_visitante":
        e_visitante["valor_plantilla"],


        "champions_local":
        e_local["jugadores_champions"],


        "champions_visitante":
        e_visitante["jugadores_champions"],


        "edad_local":
        e_local["edad_promedio"],


        "edad_visitante":
        e_visitante["edad_promedio"],


        "ataque_local":
        e_local["ataque"],


        "ataque_visitante":
        e_visitante["ataque"],


        "defensa_local":
        e_local["defensa"],


        "defensa_visitante":
        e_visitante["defensa"],


        "resultado":
        resultado


        })




dataset = pd.DataFrame(datos)



print(dataset.head())


print("\nTamaño:")

print(dataset.shape)



dataset.to_csv(

"data/dataset_mundial_avanzado.csv",

index=False

)



print("\nDataset creado:")
print("data/dataset_mundial_avanzado.csv")
