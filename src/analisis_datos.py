
# PROYECTO: MUNDIAL PREDICTOR ML


import pandas as pd
import os


print("\n================================")
print(" MUNDIAL PREDICTOR ML")
print(" ANALISIS DE DATOS")
print("================================\n")


# Ruta del archivo
ruta = "data/results.csv"


# Verificar que existe el archivo
if os.path.exists(ruta):

    print("Archivo encontrado correctamente\n")


    # Cargar dataset
    df = pd.read_csv(ruta)


    # Mostrar primeras filas
    print("===== PRIMEROS DATOS =====")
    print(df.head())


    # Mostrar tamaño del dataset
    print("\n===== TAMAÑO DEL DATASET =====")
    print(df.shape)


    # Mostrar columnas
    print("\n===== COLUMNAS =====")
    print(df.columns)


    # Información general
    print("\n===== INFORMACION DEL DATASET =====")
    print(df.info())


    # Estadísticas
    print("\n===== ESTADISTICAS =====")
    print(df.describe())


    # Valores faltantes
    print("\n===== DATOS FALTANTES =====")
    print(df.isnull().sum())


    # Mostrar últimos partidos
    print("\n===== ULTIMOS REGISTROS =====")
    print(df.tail())


else:

    print("ERROR: No se encontró results.csv")
    print("Revisa que esté dentro de la carpeta data")
