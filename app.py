import json
from pathlib import Path

import altair as alt
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
    bt = pd.read_csv(DIR / "backtest_iueam.csv") if (DIR / "backtest_iueam.csv").exists() else None
    return modelo, hist, meta, bt


modelo, hist, meta, bt = cargar()

st.title("📈 Pronóstico de estudiantes nuevos")
st.caption("Administración de Empresas · Institución Universitaria EAM · Fuente: SNIES (MinEducación)")

# Error típico del modelo en la IUEAM (RMSE de la prueba) para la banda de incertidumbre
rmse = next((f.get("IUEAM_RMSE") for f in meta["tabla"] if f.get("modelo") == meta["modelo"]), None)

with st.sidebar:
    st.header("Parámetros")
    n = st.slider("Periodos a pronosticar", 1, 6, 4)
    ajuste = st.slider("Escenario (ajuste sobre el pronóstico, %)", -30, 30, 0,
                       help="Simula una campaña de admisiones mejor o peor que la tendencia histórica.")
    banda = st.checkbox("Mostrar banda de incertidumbre", value=True,
                        help="Pronóstico ± el error típico (RMSE) del modelo en los periodos de prueba.")
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
if rmse is not None:
    pron["inferior"] = (pron["pronostico"] - rmse).clip(lower=0)
    pron["superior"] = pron["pronostico"] + rmse

# Datos en formato largo para la gráfica (la línea del pronóstico parte del último dato real)
orden = hist["periodo"].tolist() + pron["periodo"].tolist()
ultimo = hist.iloc[-1]
partes = [
    pd.DataFrame({"periodo": hist["periodo"], "estudiantes": hist["estudiantes"], "serie": "Histórico"}),
    pd.DataFrame({"periodo": [ultimo["periodo"]] + pron["periodo"].tolist(),
                  "estudiantes": [ultimo["estudiantes"]] + pron["pronostico"].tolist(), "serie": "Pronóstico (modelo)"}),
    pd.DataFrame({"periodo": [ultimo["periodo"]] + pron["periodo"].tolist(),
                  "estudiantes": [ultimo["estudiantes"]] + pron["referencia"].tolist(),
                  "serie": "Referencia (mismo semestre año anterior)"}),
]
lineas = pd.concat(partes, ignore_index=True)

x = alt.X("periodo:N", sort=orden, title="Periodo", axis=alt.Axis(labelAngle=-60))
color = alt.Color("serie:N", title=None, legend=alt.Legend(orient="bottom"),
                  scale=alt.Scale(domain=["Histórico", "Pronóstico (modelo)", "Referencia (mismo semestre año anterior)"],
                                  range=["#4c78a8", "#e45756", "#7f7f7f"]))
capa = alt.Chart(lineas).mark_line(point=True).encode(
    x=x, y=alt.Y("estudiantes:Q", title="Estudiantes de primer curso"), color=color,
    strokeDash=alt.condition(alt.datum.serie == "Referencia (mismo semestre año anterior)", alt.value([5, 4]), alt.value([0])),
    tooltip=["periodo", "serie", "estudiantes"])
if banda and rmse is not None:
    area = alt.Chart(pron).mark_area(opacity=0.18, color="#e45756").encode(x=x, y="inferior:Q", y2="superior:Q")
    grafica = (area + capa)
else:
    grafica = capa

c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Histórico y pronóstico")
    st.altair_chart(grafica.properties(height=380), width="stretch")
with c2:
    st.subheader("Pronóstico por periodo")
    cols = ["periodo", "pronostico", "referencia"] + (["inferior", "superior"] if rmse is not None else [])
    nombres = {"periodo": "Periodo", "pronostico": "Modelo", "referencia": "Referencia",
               "inferior": "Mínimo", "superior": "Máximo"}
    st.dataframe(pron[cols].rename(columns=nombres), hide_index=True)
    st.metric("Total pronosticado (modelo)", int(pron["pronostico"].sum()))
    st.caption("Referencia = mismo semestre del año anterior. Mínimo y máximo = pronóstico ± RMSE de prueba; "
               "los pronósticos a varios periodos acumulan más incertidumbre.")

if bt is not None:
    st.subheader("¿Qué tan bien habría predicho el modelo? (periodos de prueba)")
    ordenbt = bt["periodo"].tolist()
    larga = bt.rename(columns={"real": "Real", "modelo": "Modelo", "referencia": "Referencia"}) \
              .melt("periodo", var_name="serie", value_name="estudiantes")
    gbt = alt.Chart(larga).mark_line(point=True).encode(
        x=alt.X("periodo:N", sort=ordenbt, title="Periodo"),
        y=alt.Y("estudiantes:Q", title="Estudiantes de primer curso"),
        color=alt.Color("serie:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=["Real", "Modelo", "Referencia"], range=["#4c78a8", "#e45756", "#7f7f7f"])),
        tooltip=["periodo", "serie", "estudiantes"]).properties(height=300)
    st.altair_chart(gbt, width="stretch")
    st.caption("El modelo se entrenó con datos hasta 2022 y aquí se compara con lo que realmente ocurrió en 2023-2025.")

with st.expander("Desempeño del modelo (periodos de prueba)"):
    st.dataframe(pd.DataFrame(meta["tabla"]))
    st.caption("MAE/RMSE en número de estudiantes; MAPE en %; Sesgo = predicho − real (positivo = sobreestima).")
