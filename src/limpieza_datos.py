# ==========================================
# MUNDIAL PREDICTOR ML
# PASO 4: LIMPIEZA Y CREACION DE VARIABLES
# ==========================================


import pandas as pd


print("\n==============================")
print(" LIMPIEZA DE DATOS")
print("==============================\n")


# Cargar datos

df = pd.read_csv("data/results.csv")


print("Datos originales:")
print(df.head())


# ==========================================
# ELIMINAR COLUMNAS QUE NO NECESITAMOS
# ==========================================


columnas = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score"
]


df = df[columnas]


# eliminar valores vacios

df = df.dropna()


# ==========================================
# CREAR RESULTADO DEL PARTIDO
# ==========================================

def resultado(row):

    if row["home_score"] > row["away_score"]:
        return 2   # gana local

    elif row["home_score"] < row["away_score"]:
        return 0   # gana visitante

    else:
        return 1   # empate



df["resultado"] = df.apply(resultado, axis=1)



# ==========================================
# CREAR DIFERENCIA DE GOLES
# ==========================================


df["diferencia_goles"] = (
    df["home_score"] -
    df["away_score"]
)



# ==========================================
# CREAR VARIABLES PARA ML
# ==========================================


df_ml = pd.DataFrame()


df_ml["goles_local"] = df["home_score"]

df_ml["goles_visitante"] = df["away_score"]

df_ml["diferencia_goles"] = df["diferencia_goles"]

df_ml["resultado"] = df["resultado"]



# Mostrar nuevo dataset


print("\nDataset preparado para ML:")
print(df_ml.head())


print("\nDimensiones:")
print(df_ml.shape)



# Guardar nuevo archivo

df_ml.to_csv(
    "data/dataset_ml.csv",
    index=False
)



print("\nArchivo creado:")
print("data/dataset_ml.csv")
