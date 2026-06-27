# ==========================================
# MUNDIAL PREDICTOR ML
# PREDICTOR FINAL
# ==========================================


import pandas as pd

import joblib



print("\n==============================")
print(" PREDICTOR DEL MUNDIAL")
print("==============================\n")



# ==========================================
# CARGAR EQUIPOS
# ==========================================


equipos = pd.read_csv(

"data/equipos_mundial.csv"

)



# convertir a diccionario

equipos_dict = equipos.set_index(

"equipo"

).to_dict(

"index"

)



# ==========================================
# CARGAR MODELO
# ==========================================


modelo = joblib.load(

"models/XGBoost_avanzado.pkl"

)



# ==========================================
# INGRESAR PARTIDO
# ==========================================


local = input(
"Equipo local: "
)


visitante = input(
"Equipo visitante: "
)



# verificar equipos


if local not in equipos_dict:

    print("Equipo local no existe")

    exit()



if visitante not in equipos_dict:

    print("Equipo visitante no existe")

    exit()




e_local = equipos_dict[local]

e_visitante = equipos_dict[visitante]



# ==========================================
# CREAR DATOS DEL PARTIDO
# ==========================================


partido = pd.DataFrame([{


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
e_visitante["defensa"]

}])



# ==========================================
# PREDICCION
# ==========================================


resultado = modelo.predict(

partido

)



probabilidad = modelo.predict_proba(

partido

)[0]




print("\n==============================")

print(" RESULTADO")

print("==============================")



if resultado[0] == 0:


    print(
    "Gana:",
    visitante
    )


elif resultado[0] == 1:


    print(
    "Empate"
    )


else:


    print(
    "Gana:",
    local
    )



print("\nProbabilidades:")



print(

local,

":",

round(probabilidad[2]*100,2),

"%"

)



print(

"Empate:",

round(probabilidad[1]*100,2),

"%"

)



print(

visitante,

":",

round(probabilidad[0]*100,2),

"%"

)
