import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from forecast_utils import recursive_forecast

st.set_page_config(page_title="Pronóstico de estudiantes nuevos – IUEAM", page_icon="📈", layout="wide")
DIR = Path(__file__).parent


@st.cache_resource
def cargar():
    modelo = joblib.load(DIR / "modelo.joblib")
    hist = pd.read_csv(DIR / "historico_iueam.csv")
    meta = json.load(open(DIR / "metricas.json", encoding="utf-8"))
    return modelo, hist, meta


modelo, hist, meta = cargar()

st.title("📈 Pronóstico de estudiantes nuevos")
st.caption("Administración de Empresas · Institución Universitaria EAM · Fuente: SNIES (MinEducación)")

with st.sidebar:
    st.header("Parámetros")
    n = st.slider("Periodos a pronosticar", 1, 6, 4)
    ajuste = st.slider("Escenario (ajuste sobre el pronóstico, %)", -30, 30, 0,
                       help="Simula una campaña de admisiones mejor o peor que la tendencia histórica.")
    st.markdown(f"**Modelo:** {meta['modelo']}")

pron = recursive_forecast(modelo, hist["estudiantes"].tolist(), int(hist["idx"].max()), n)
pron["pronostico"] = (pron["pronostico"] * (1 + ajuste / 100)).round(0)
pron["periodo"] = pron["anio"].astype(str) + "-" + pron["semestre"].astype(str)
hist["periodo"] = hist["anio"].astype(str) + "-" + hist["semestre"].astype(str)

# Referencia simple: repetir el valor del mismo semestre del año anterior
serie = dict(zip(hist["idx"], hist["estudiantes"]))
ref = []
for i in pron["idx"]:
    serie[i] = serie[i - 2]
    ref.append(serie[i])
pron["referencia"] = ref

graf = pd.concat([
    hist.set_index("periodo")["estudiantes"].rename("Histórico"),
    pron.set_index("periodo")["pronostico"].rename("Pronóstico (modelo)"),
    pron.set_index("periodo")["referencia"].rename("Referencia (mismo semestre año anterior)"),
], axis=1)

c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Histórico y pronóstico")
    st.line_chart(graf)
with c2:
    st.subheader("Pronóstico por periodo")
    st.dataframe(pron[["periodo", "pronostico", "referencia"]].rename(
                     columns={"periodo": "Periodo", "pronostico": "Modelo", "referencia": "Referencia"}),
                 hide_index=True)
    st.metric("Total pronosticado (modelo)", int(pron["pronostico"].sum()))
    st.caption("Referencia = mismo semestre del año anterior. En series cortas como esta suele ser muy competitiva.")

with st.expander("Desempeño del modelo (periodos de prueba)"):
    st.dataframe(pd.DataFrame(meta["tabla"]))
    st.caption("MAE/RMSE en número de estudiantes; MAPE en %. Los pronósticos a varios periodos acumulan incertidumbre.")
