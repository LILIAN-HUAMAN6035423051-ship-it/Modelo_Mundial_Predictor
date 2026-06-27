import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ==========================================
# CONFIGURACION
# ==========================================

st.set_page_config(
    page_title="Mundial Predictor ML",
    page_icon="⚽",
    layout="wide"
)


st.title("⚽ Predictor Mundial ML")

st.write(
    "Sistema de predicción de partidos usando Machine Learning"
)



# ==========================================
# CARGAR DATOS
# ==========================================


equipos = pd.read_csv(
    "data/equipos_mundial.csv"
)


equipos = equipos.drop_duplicates(
    subset="equipo"
)


lista_equipos = equipos["equipo"].tolist()



equipos_dict = (
    equipos
    .set_index("equipo")
    .to_dict("index")
)



modelo = joblib.load(
    "models/XGBoost_avanzado.pkl"
)



# ==========================================
# SELECTORES
# ==========================================


st.sidebar.header(
    "⚽ Seleccionar partido"
)



local = st.sidebar.selectbox(

    "Equipo local",

    lista_equipos,

    index=0

)



visitante = st.sidebar.selectbox(

    "Equipo visitante",

    lista_equipos,

    index=1

)



# ==========================================
# DATOS ACTUALES
# ==========================================


datos_local = equipos_dict[local]


datos_visitante = equipos_dict[visitante]



# ==========================================
# MOSTRAR EQUIPOS
# ==========================================


col1,col2 = st.columns(2)



with col1:


    st.subheader(
        "Equipo local"
    )


    st.markdown(
        f"## 🏳️ {local}"
    )


    st.metric(
        "Clasificación FIFA",
        datos_local["ranking_fifa"]
    )


    st.metric(
        "Valor plantilla",
        datos_local["valor_plantilla"]
    )


    st.metric(
        "Ataque",
        datos_local["ataque"]
    )


    st.metric(
        "Defensa",
        datos_local["defensa"]
    )





with col2:


    st.subheader(
        "Equipo visitante"
    )


    st.markdown(
        f"## 🏳️ {visitante}"
    )


    st.metric(
        "Clasificación FIFA",
        datos_visitante["ranking_fifa"]
    )


    st.metric(
        "Valor plantilla",
        datos_visitante["valor_plantilla"]
    )


    st.metric(
        "Ataque",
        datos_visitante["ataque"]
    )


    st.metric(
        "Defensa",
        datos_visitante["defensa"]
    )




# ==========================================
# PREDICCION
# ==========================================


st.divider()



if st.button(
    "🔮 Predecir partido"
):


    partido = pd.DataFrame([{


        "ranking_local":
        datos_local["ranking_fifa"],


        "ranking_visitante":
        datos_visitante["ranking_fifa"],


        "valor_local":
        datos_local["valor_plantilla"],


        "valor_visitante":
        datos_visitante["valor_plantilla"],


        "champions_local":
        datos_local["jugadores_champions"],


        "champions_visitante":
        datos_visitante["jugadores_champions"],


        "edad_local":
        datos_local["edad_promedio"],


        "edad_visitante":
        datos_visitante["edad_promedio"],


        "ataque_local":
        datos_local["ataque"],


        "ataque_visitante":
        datos_visitante["ataque"],


        "defensa_local":
        datos_local["defensa"],


        "defensa_visitante":
        datos_visitante["defensa"]


    }])



    pred = modelo.predict(

        partido

    )[0]



    prob = modelo.predict_proba(

        partido

    )[0]



    st.header(
        "Resultado"
    )



    if pred == 2:


        st.success(
            f"🏆 Ganador probable: {local}"
        )


    elif pred == 0:


        st.success(
            f"🏆 Ganador probable: {visitante}"
        )


    else:


        st.warning(
            "⚖️ Empate"
        )




    resultados = pd.DataFrame({


        "Equipo":[

            visitante,

            "Empate",

            local

        ],


        "Probabilidad":[

            prob[0]*100,

            prob[1]*100,

            prob[2]*100

        ]

    })



    st.subheader(
        "Probabilidades"
    )


    st.bar_chart(

        resultados.set_index(
            "Equipo"
        )

    )



# ==========================================
# MONTE CARLO
# ==========================================


st.divider()



st.header(
    "🏆 Probabilidad de campeón"
)



try:


    campeones = pd.read_csv(

        "data/probabilidad_campeon.csv"

    )



    campeones.columns=[

        "Equipo",

        "Probabilidad"

    ]



    st.dataframe(

        campeones,

        use_container_width=True

    )



    fig,ax = plt.subplots()


    ax.bar(

        campeones["Equipo"],

        campeones["Probabilidad"]

    )



    plt.xticks(

        rotation=45

    )


    st.pyplot(fig)



except:


    st.info(
        "Ejecuta montecarlo_mundial.py primero"
    )