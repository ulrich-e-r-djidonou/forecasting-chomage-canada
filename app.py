"""
Dashboard Streamlit — Prévision du taux de chômage canadien
Exécuter avec : streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# ============================================================
# Configuration
# ============================================================
st.set_page_config(
    page_title="Prévision du chômage — Canada",
    page_icon="📊",
    layout="wide"
)

DATA_DIR = Path(__file__).parent / "data" / "processed"

PROVINCES_FR = {
    "Alberta": "Alberta",
    "British Columbia": "Colombie-Britannique",
    "Manitoba": "Manitoba",
    "New Brunswick": "Nouveau-Brunswick",
    "Newfoundland and Labrador": "Terre-Neuve-et-Labrador",
    "Nova Scotia": "Nouvelle-Écosse",
    "Ontario": "Ontario",
    "Prince Edward Island": "Île-du-Prince-Édouard",
    "Quebec": "Québec",
    "Saskatchewan": "Saskatchewan",
    "Canada": "Canada"
}

# ============================================================
# Chargement des données
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "chomage_mensuel.csv", parse_dates=["date"])
    return df

@st.cache_data
def load_comparison():
    try:
        return pd.read_csv(DATA_DIR / "comparaison_modeles.csv")
    except FileNotFoundError:
        return None

# ============================================================
# Modèles
# ============================================================
@st.cache_data
def run_arima(series, order, horizon):
    model = ARIMA(series, order=order).fit()
    fc = model.get_forecast(steps=horizon)
    summary = fc.summary_frame(alpha=0.05)
    return summary

@st.cache_data
def run_prophet(df_prophet, horizon):
    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=False,
        daily_seasonality=False, changepoint_prior_scale=0.05
    )
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)
    return forecast

# ============================================================
# Interface
# ============================================================
st.title("Prévision du taux de chômage au Canada")
st.markdown(
    "Données : Statistique Canada, Table 14-10-0287-03 | "
    "Licence du gouvernement ouvert — Canada"
)

df = load_data()

if df is None or df.empty:
    st.error("Données non trouvées. Exécutez les notebooks 01 et 02 d'abord.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Paramètres")

geo_options = {PROVINCES_FR.get(g, g): g for g in sorted(df["geo"].unique())}
geo_label = st.sidebar.selectbox("Géographie", list(geo_options.keys()), index=0)
geo = geo_options[geo_label]

horizon = st.sidebar.slider("Horizon de prévision (mois)", 3, 24, 12)

model_choice = st.sidebar.multiselect(
    "Modèles", ["ARIMA", "Prophet"], default=["ARIMA", "Prophet"]
)

show_history = st.sidebar.slider("Historique affiché (années)", 1, 20, 5)

# --- Données pour la géographie sélectionnée ---
df_geo = df[df["geo"] == geo].sort_values("date").reset_index(drop=True)
series = df_geo.set_index("date")["unemployment_rate"].asfreq("MS")

# ============================================================
# Graphique principal
# ============================================================
st.header(f"Taux de chômage — {geo_label}")

fig = go.Figure()

# Historique
hist_start = series.index.max() - pd.DateOffset(years=show_history)
hist = series[series.index >= hist_start]
fig.add_trace(go.Scatter(
    x=hist.index, y=hist.values,
    name="Historique", line=dict(color="#2c3e50", width=2)
))

# ARIMA
if "ARIMA" in model_choice:
    with st.spinner("ARIMA en cours..."):
        # Sélection rapide d'ordre
        best_aic = np.inf
        best_order = (1, 1, 1)
        for p in range(4):
            for d in range(2):
                for q in range(4):
                    if p == 0 and q == 0:
                        continue
                    try:
                        m = ARIMA(series, order=(p, d, q)).fit()
                        if m.aic < best_aic:
                            best_aic = m.aic
                            best_order = (p, d, q)
                    except:
                        continue

        arima_fc = run_arima(series, best_order, horizon)

        fig.add_trace(go.Scatter(
            x=arima_fc.index, y=arima_fc["mean"],
            name=f"ARIMA{best_order}",
            line=dict(color="#3498db", width=2, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=list(arima_fc.index) + list(arima_fc.index[::-1]),
            y=list(arima_fc["mean_ci_upper"]) + list(arima_fc["mean_ci_lower"][::-1]),
            fill="toself", fillcolor="rgba(52,152,219,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="IC 95% ARIMA", showlegend=False
        ))

# Prophet
if "Prophet" in model_choice:
    with st.spinner("Prophet en cours..."):
        df_prophet = df_geo[["date", "unemployment_rate"]].copy()
        df_prophet.columns = ["ds", "y"]

        prophet_fc = run_prophet(df_prophet, horizon)
        fc_future = prophet_fc.tail(horizon)

        fig.add_trace(go.Scatter(
            x=fc_future["ds"], y=fc_future["yhat"],
            name="Prophet",
            line=dict(color="#e74c3c", width=2, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=list(fc_future["ds"]) + list(fc_future["ds"][::-1]),
            y=list(fc_future["yhat_upper"]) + list(fc_future["yhat_lower"][::-1]),
            fill="toself", fillcolor="rgba(231,76,60,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="IC 95% Prophet", showlegend=False
        ))

fig.update_layout(
    yaxis_title="Taux de chômage (%)",
    hovermode="x unified",
    template="plotly_white",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Statistiques clés
# ============================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dernier taux observé", f"{series.iloc[-1]:.1f}%")
col2.metric("Date", series.index[-1].strftime("%Y-%m"))

if "ARIMA" in model_choice:
    arima_last = arima_fc["mean"].iloc[-1]
    delta_a = arima_last - series.iloc[-1]
    col3.metric(f"ARIMA à {horizon} mois", f"{arima_last:.1f}%", f"{delta_a:+.1f} pp")

if "Prophet" in model_choice:
    prophet_last = fc_future["yhat"].iloc[-1]
    delta_p = prophet_last - series.iloc[-1]
    col4.metric(f"Prophet à {horizon} mois", f"{prophet_last:.1f}%", f"{delta_p:+.1f} pp")

# ============================================================
# Tableau de comparaison (si disponible)
# ============================================================
df_comp = load_comparison()
if df_comp is not None:
    st.header("Comparaison des modèles (ensemble test)")
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

# ============================================================
# Toutes les provinces
# ============================================================
st.header("Comparaison interprovinciale")

fig_all = go.Figure()
for prov in sorted(df["geo"].unique()):
    subset = df[df["geo"] == prov]
    fig_all.add_trace(go.Scatter(
        x=subset["date"], y=subset["unemployment_rate"],
        name=PROVINCES_FR.get(prov, prov),
        visible=True if prov == "Canada" else "legendonly"
    ))

fig_all.update_layout(
    yaxis_title="Taux de chômage (%)",
    hovermode="x unified",
    template="plotly_white",
    height=500
)
st.plotly_chart(fig_all, use_container_width=True)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    "Source : Statistique Canada, Table 14-10-0287-03 | "
    "Licence du gouvernement ouvert — Canada | "
    "Modèles : ARIMA (statsmodels) + Prophet (Meta)"
)
