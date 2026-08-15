import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

RUTA_DATOS = "data/autos_limpio.csv"


@st.cache_data
def cargar_datos():
    return pd.read_csv(RUTA_DATOS)


st.title("Eficiencia de autos (mpg)")

df = cargar_datos()

origenes = ["todos"] + sorted(df["origin"].unique())
origen = st.sidebar.selectbox("Origen", origenes)

peso_min, peso_max = int(df["weight"].min()), int(df["weight"].max())
rango_peso = st.sidebar.slider(
    "Peso (weight)",
    min_value=peso_min,
    max_value=peso_max,
    value=(peso_min, peso_max),
)

filtrado = df[(df["weight"] >= rango_peso[0]) & (df["weight"] <= rango_peso[1])]
if origen != "todos":
    filtrado = filtrado[filtrado["origin"] == origen]

if filtrado.empty:
    st.warning("No hay autos que cumplan estos filtros.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Autos filtrados", len(filtrado))
col2.metric("mpg promedio", f"{filtrado['mpg'].mean():.2f}")
col3.metric("mpg mediana", f"{filtrado['mpg'].median():.2f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(filtrado["weight"], filtrado["mpg"], alpha=0.6, edgecolor="none")
ax.set_xlabel("weight")
ax.set_ylabel("mpg")
ax.set_title("Relacion peso vs mpg")
st.pyplot(fig)
