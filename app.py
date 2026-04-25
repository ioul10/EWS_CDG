"""
app.py
──────
EWS-CDG — Early Warning System
Module 1 : Construction de Portefeuille MASI20

Streamlit interface for portfolio construction with:
  - Manual weight allocation
  - Markowitz optimization (3 objectives)
  - Efficient frontier visualization
  - Export to CSV

Author  : [Your Name]
Project : Early Warning System — CDG
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import io

# Local modules
import sys
import os
sys.path.append(os.path.dirname(__file__))

from data.masi20_tickers import MASI20_UNIVERSE, get_display_label, get_sector, SECTOR_COLORS
from modules.data_loader  import fetch_prices, fetch_index, compute_simple_returns, validate_date_range, get_data_summary
from modules.portfolio    import Portfolio, normalize_weights, equal_weights
from modules.markowitz    import MarkowitzOptimizer
from modules.ews_statistical import EWSStatistical


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EWS · CDG — Construction de Portefeuille",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        color: #1F4E79;
        letter-spacing: -0.5px;
        margin-bottom: 0;
    }

    .sub-title {
        font-size: 0.95rem;
        color: #5B7FA6;
        margin-top: 0.2rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        color: #1F4E79;
        border-left: 4px solid #2E75B6;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #f8fbff 0%, #edf4fd 100%);
        border: 1px solid #d0e4f7;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #5B7FA6;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        color: #1F4E79;
        font-weight: 400;
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .badge-success { background: #e8f5e9; color: #2e7d32; }
    .badge-warning { background: #fff8e1; color: #f57f17; }
    .badge-info    { background: #e3f2fd; color: #1565c0; }

    .divider {
        border: none;
        border-top: 1px solid #e0ecf8;
        margin: 1.5rem 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1F4E79 0%, #0d2d4a 100%);
    }
    [data-testid="stSidebar"] * {
        color: #cce0f5 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff !important;
        font-family: 'DM Serif Display', serif;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #163a5c, #1F4E79);
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(31,78,121,0.3);
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────

col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<div class="main-title">📊 Système d\'Alerte Précoce · CDG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">MODULE 1 — CONSTRUCTION DE PORTEFEUILLE MASI20</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Sidebar — Configuration ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    # Date range
    st.markdown("**Période d'analyse**")
    start_date = st.date_input("Date de début", value=date(2019, 1, 1), min_value=date(2010, 1, 1))
    end_date   = st.date_input("Date de fin",   value=date.today())

    st.markdown("---")

    # Initial value
    st.markdown("**Valeur initiale du portefeuille**")
    initial_value = st.number_input(
        "Montant (MAD)",
        min_value=1_000_000,
        max_value=100_000_000_000,
        value=100_000_000,
        step=1_000_000,
        format="%d",
    )
    st.caption(f"≈ {initial_value/1_000_000:.0f} M MAD")

    st.markdown("---")

    # Weight mode
    st.markdown("**Mode d'allocation**")
    weight_mode = st.radio(
        "",
        options=["Manuel", "Markowitz"],
        index=1,
        label_visibility="collapsed",
    )

    if weight_mode == "Markowitz":
        objective = st.selectbox(
            "Objectif d'optimisation",
            options=["Maximiser le ratio de Sharpe", "Minimiser la variance", "Cibler un rendement"],
        )
        if objective == "Cibler un rendement":
            target_return = st.slider("Rendement cible (%)", 1, 25, 8) / 100
        max_weight = st.slider("Poids maximum par titre (%)", 10, 50, 40) / 100

    st.markdown("---")
    st.markdown("**Contraintes**")
    risk_free = st.slider("Taux sans risque (%)", 0, 10, 3) / 100


# ─── Ticker Selection ─────────────────────────────────────────────────────────

st.markdown('<div class="section-header">① Sélection des titres</div>', unsafe_allow_html=True)

# Group tickers by sector for display
sectors = {}
for ticker, info in MASI20_UNIVERSE.items():
    sector = info["sector"]
    if sector not in sectors:
        sectors[sector] = []
    sectors[sector].append(ticker)

selected_tickers = []
cols = st.columns(3)
col_idx = 0

for sector, tickers in sectors.items():
    with cols[col_idx % 3]:
        st.markdown(f"**{sector}**")
        for ticker in tickers:
            label = f"{MASI20_UNIVERSE[ticker]['name']} `{ticker}`"
            if st.checkbox(label, value=True, key=f"chk_{ticker}"):
                selected_tickers.append(ticker)
    col_idx += 1

# Validation
if len(selected_tickers) < 2:
    st.warning("⚠️ Veuillez sélectionner au moins 2 titres.")
    st.stop()

st.markdown(f"<span class='status-badge badge-info'>✓ {len(selected_tickers)} titres sélectionnés</span>", unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Manual Weights ───────────────────────────────────────────────────────────

manual_weights = {}
if weight_mode == "Manuel":
    st.markdown('<div class="section-header">② Allocation manuelle des poids</div>', unsafe_allow_html=True)

    equal_w = 100 / len(selected_tickers)
    total_pct = 0
    cols_w = st.columns(min(4, len(selected_tickers)))

    for i, ticker in enumerate(selected_tickers):
        with cols_w[i % len(cols_w)]:
            w = st.number_input(
                f"{MASI20_UNIVERSE[ticker]['name']}",
                min_value=0.0, max_value=100.0,
                value=round(equal_w, 1), step=0.1,
                key=f"w_{ticker}",
            )
            manual_weights[ticker] = w / 100
            total_pct += w

    delta = abs(total_pct - 100)
    if delta > 0.1:
        st.error(f"❌ La somme des poids est {total_pct:.1f}% — elle doit être égale à 100%.")
        st.stop()
    else:
        st.success(f"✅ Somme des poids : {total_pct:.1f}%")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Build Button ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">③ Construction du portefeuille</div>', unsafe_allow_html=True)

build_col, _ = st.columns([2, 5])
with build_col:
    build = st.button("🚀 Construire le portefeuille", use_container_width=True)

if not build:
    st.info("👆 Configurez vos paramètres dans la barre latérale, sélectionnez vos titres, puis cliquez sur **Construire le portefeuille**.")
    st.stop()


# ─── Data Loading ─────────────────────────────────────────────────────────────

with st.spinner("📡 Chargement des données..."):
    try:
        validate_date_range(str(start_date), str(end_date))
        prices, valid_tickers, sources = fetch_prices(
            selected_tickers,
            start=str(start_date),
            end=str(end_date),
            progress=False,
        )
        masi = fetch_index(start=str(start_date), end=str(end_date))
    except ValueError as e:
        st.error(f"❌ {e}")
        st.markdown("""
        **Comment résoudre :**
        1. Allez sur [fr.investing.com](https://fr.investing.com)
        2. Cherchez le titre (ex: *Attijariwafa Bank*)
        3. Cliquez sur **Données Historiques** → **Télécharger**
        4. Renommez le fichier : `ATW.csv`, `IAM.csv`, etc.
        5. Placez-le dans le dossier `data/prices/`
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur inattendue : {e}")
        st.stop()

if len(valid_tickers) < 2:
    st.error("❌ Moins de 2 titres chargés. Ajoutez des fichiers CSV dans data/prices/")
    st.stop()

# Show data sources
csv_tickers   = [t for t, s in sources.items() if s == "csv"]
yahoo_tickers = [t for t, s in sources.items() if s == "yahoo"]
failed_tickers= [t for t, s in sources.items() if s == "failed"]

if csv_tickers:
    st.success(f"✅ {len(csv_tickers)} titre(s) chargés depuis CSV local")
if yahoo_tickers:
    st.info(f"📡 {len(yahoo_tickers)} titre(s) chargés depuis Yahoo Finance")
if failed_tickers:
    st.warning(f"⚠️ Titres non disponibles : {', '.join(failed_tickers)}")


# ─── Optimization or Manual ───────────────────────────────────────────────────

returns_df = compute_simple_returns(prices)

with st.spinner("⚙️ Calcul des poids optimaux..." if weight_mode == "Markowitz" else "⚙️ Construction en cours..."):
    if weight_mode == "Markowitz":
        optimizer = MarkowitzOptimizer(
            returns=returns_df,
            risk_free_rate=risk_free,
            max_weight=max_weight,
        )
        if objective == "Maximiser le ratio de Sharpe":
            opt_result = optimizer.maximize_sharpe()
        elif objective == "Minimiser la variance":
            opt_result = optimizer.minimize_variance()
        else:
            opt_result = optimizer.target_return(target_return)

        weights_dict = opt_result["weights"]
        frontier_df  = optimizer.efficient_frontier()
        mc_df        = optimizer.monte_carlo_frontier()
    else:
        weights_dict = normalize_weights({t: manual_weights[t] for t in valid_tickers if t in manual_weights})
        opt_result   = None
        frontier_df  = None
        mc_df        = None

    # Build portfolio
    portfolio = Portfolio(
        prices=prices,
        weights=weights_dict,
        initial_value=initial_value,
        name="Portefeuille MASI20 — CDG",
    )

st.success("✅ Portefeuille construit avec succès.")
# ── EWS Statistical Layer ──────────────────────────────
ews = EWSStatistical(
    returns=portfolio.returns,
    total_value=portfolio.total_value,
)
current = ews.get_current_status()
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Results ──────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">④ Résultats</div>', unsafe_allow_html=True)

# KPI cards
summary = portfolio.get_summary()
kpi_cols = st.columns(4)
kpis = [
    ("Rendement annualisé",   summary["Rendement annualisé"],   "📈"),
    ("Volatilité annualisée", summary["Volatilité annualisée"], "📉"),
    ("Ratio de Sharpe",       summary["Ratio de Sharpe"],        "⚡"),
    ("Drawdown maximum",      summary["Drawdown maximum"],       "🔻"),
]
for col, (label, value, icon) in zip(kpi_cols, kpis):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# Valeur initiale / finale
val_cols = st.columns(2)
with val_cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">💰 Valeur initiale</div>
        <div class="metric-value">{summary['Valeur initiale (MAD)']}</div>
    </div>""", unsafe_allow_html=True)
with val_cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">💼 Valeur finale</div>
        <div class="metric-value">{summary['Valeur finale (MAD)']}</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Charts ───────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Évolution", "🥧 Allocation", "📊 Frontière efficiente", "📋 Données", "🚨 Alertes EWS"])
# Tab 1 — Portfolio Value Evolution
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio.total_value.index,
        y=portfolio.total_value.values,
        mode="lines",
        name=portfolio.name,
        line=dict(color="#1F4E79", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(31,78,121,0.08)",
    ))

    # Normalize MASI to same initial value for comparison
    masi_aligned = masi.reindex(portfolio.total_value.index).ffill()
    masi_normalized = masi_aligned / masi_aligned.iloc[0] * initial_value
    fig.add_trace(go.Scatter(
        x=masi_normalized.index,
        y=masi_normalized.values,
        mode="lines",
        name="MASI (normalisé)",
        line=dict(color="#ED7D31", width=1.5, dash="dash"),
    ))

    fig.update_layout(
        title="Évolution de la valeur du portefeuille vs MASI",
        xaxis_title="Date",
        yaxis_title="Valeur (MAD)",
        template="plotly_white",
        height=450,
        font=dict(family="DM Sans", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 2 — Allocation
with tab2:
    weights_series = pd.Series(weights_dict)
    labels = [f"{MASI20_UNIVERSE[t]['name']}" for t in weights_series.index]
    colors = [SECTOR_COLORS.get(get_sector(t), "#999999") for t in weights_series.index]

    fig_pie = go.Figure(go.Pie(
        labels=labels,
        values=weights_series.values,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+percent",
        textfont=dict(family="DM Sans", size=12),
    ))
    fig_pie.update_layout(
        title=f"Allocation du portefeuille — {weight_mode}",
        template="plotly_white",
        height=500,
        font=dict(family="DM Sans"),
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Weights table
    w_df = pd.DataFrame({
        "Titre": labels,
        "Ticker": list(weights_series.index),
        "Secteur": [get_sector(t) for t in weights_series.index],
        "Poids (%)": (weights_series.values * 100).round(2),
        "Capital alloué (MAD)": (weights_series.values * initial_value).round(0).astype(int),
    }).sort_values("Poids (%)", ascending=False).reset_index(drop=True)
    st.dataframe(w_df, use_container_width=True, hide_index=True)

# Tab 3 — Efficient Frontier
with tab3:
    if frontier_df is not None and mc_df is not None:
        fig_ef = go.Figure()

        # Monte Carlo cloud
        fig_ef.add_trace(go.Scatter(
            x=mc_df["Volatilité"] * 100,
            y=mc_df["Rendement"] * 100,
            mode="markers",
            marker=dict(
                size=3,
                color=mc_df["Sharpe"],
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Sharpe"),
                opacity=0.5,
            ),
            name="Portefeuilles simulés",
        ))

        # Efficient frontier line
        fig_ef.add_trace(go.Scatter(
            x=frontier_df["Volatilité"] * 100,
            y=frontier_df["Rendement"] * 100,
            mode="lines",
            line=dict(color="#1F4E79", width=3),
            name="Frontière efficiente",
        ))

        # Optimal point
        fig_ef.add_trace(go.Scatter(
            x=[opt_result["volatility"] * 100],
            y=[opt_result["expected_return"] * 100],
            mode="markers",
            marker=dict(color="#ED7D31", size=14, symbol="star"),
            name=f"Portefeuille optimal ({opt_result['label']})",
        ))

        fig_ef.update_layout(
            title="Frontière Efficiente de Markowitz — MASI20",
            xaxis_title="Volatilité annualisée (%)",
            yaxis_title="Rendement annualisé (%)",
            template="plotly_white",
            height=500,
            font=dict(family="DM Sans", size=13),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ef, use_container_width=True)

        # Optimal portfolio stats
        if opt_result:
            st.markdown("**Portefeuille optimal**")
            opt_cols = st.columns(3)
            with opt_cols[0]:
                st.metric("Rendement espéré", f"{opt_result['expected_return']*100:.2f}%")
            with opt_cols[1]:
                st.metric("Volatilité",        f"{opt_result['volatility']*100:.2f}%")
            with opt_cols[2]:
                st.metric("Ratio de Sharpe",   f"{opt_result['sharpe_ratio']:.2f}")
    else:
        st.info("La frontière efficiente est disponible uniquement en mode Markowitz.")

# Tab 4 — Raw Data
with tab4:
    st.markdown("**Cours de clôture ajustés**")
    st.dataframe(prices.tail(30).style.format("{:.2f}"), use_container_width=True)

    st.markdown("**Rendements journaliers**")
    st.dataframe(returns_df.tail(30).style.format("{:.4%}"), use_container_width=True)

    # Export
    st.markdown("---")
    csv_data = portfolio.total_value.reset_index()
    csv_data.columns = ["Date", "Valeur_Totale_MAD"]

    buffer = io.BytesIO()
    csv_data.to_csv(buffer, index=False)
    st.download_button(
        label="⬇️ Exporter la valeur du portefeuille (CSV)",
        data=buffer.getvalue(),
        file_name=f"portefeuille_masi20_{date.today()}.csv",
        mime="text/csv",
    )
# Tab 5 — EWS Alertes
with tab5:

    # ── Statut du jour ────────────────────────────────
    st.markdown("### Statut du jour")

    score = current["score_total"]
    couleur = "#2e7d32" if score <= 2 else "#f57f17" if score <= 5 else "#c62828"

    st.markdown(f"""
    <div style="background:{couleur}22; border-left:6px solid {couleur};
                padding:1rem 1.5rem; border-radius:8px; margin-bottom:1rem;">
        <span style="font-size:1.5rem; font-weight:700; color:{couleur};">
            {current['niveau_alerte']}
        </span>
        <span style="color:#555; margin-left:1rem;">
            Score : {score} / 8 — {current['date']}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Score par indicateur ──────────────────────────
    st.markdown("### Détail des scores")
    c1, c2, c3, c4 = st.columns(4)

    def score_card(col, label, value, score):
        color = "#2e7d32" if score == 0 else "#f57f17" if score == 1 else "#c62828"
        badge = ["🟢 Normal", "🟡 Vigilance", "🔴 Alerte"][score]
        col.markdown(f"""
        <div style="border:1px solid {color}44; border-radius:10px;
                    padding:0.8rem; text-align:center; background:{color}11;">
            <div style="font-size:0.75rem; color:#666; font-weight:600;
                        text-transform:uppercase; letter-spacing:0.5px;">{label}</div>
            <div style="font-size:1.4rem; font-weight:700; color:{color};
                        margin:0.3rem 0;">{value}</div>
            <div style="font-size:0.75rem; color:{color};">{badge}</div>
        </div>
        """, unsafe_allow_html=True)

    score_card(c1, "Volatilité 30j",  current["volatilite"],  current["score_vol"])
    score_card(c2, "Z-Score",         current["z_score"],      current["score_z"])
    score_card(c3, "Drawdown",        current["drawdown"],     current["score_dd"])
    score_card(c4, "VaR 99%",         current["var_99"],       current["score_var"])

    st.markdown("")

    # ── Évolution du Score Total ──────────────────────
    st.markdown("### Évolution du Score d'alerte")

    ews_plot = ews.results[["Score_Total"]].dropna()

    fig_ews = go.Figure()

    # Color zones
    fig_ews.add_hrect(y0=0, y1=2, fillcolor="#2e7d32", opacity=0.06, line_width=0)
    fig_ews.add_hrect(y0=3, y1=5, fillcolor="#f57f17", opacity=0.06, line_width=0)
    fig_ews.add_hrect(y0=6, y1=8, fillcolor="#c62828", opacity=0.06, line_width=0)

    fig_ews.add_trace(go.Scatter(
        x=ews_plot.index,
        y=ews_plot["Score_Total"],
        mode="lines",
        name="Score EWS",
        line=dict(color="#1F4E79", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(31,78,121,0.08)",
    ))

    fig_ews.add_hline(y=3, line=dict(color="#f57f17", dash="dash", width=1.5),
                      annotation_text="Vigilance", annotation_position="right")
    fig_ews.add_hline(y=6, line=dict(color="#c62828", dash="dash", width=1.5),
                      annotation_text="Critique", annotation_position="right")

    fig_ews.update_layout(
        xaxis_title="Date",
        yaxis_title="Score Total (0–8)",
        yaxis=dict(range=[0, 8.5]),
        template="plotly_white",
        height=400,
        font=dict(family="DM Sans", size=13),
        hovermode="x unified",
    )
    st.plotly_chart(fig_ews, use_container_width=True)

    # ── Périodes de stress ────────────────────────────
    st.markdown("### Périodes de stress détectées")
    stress_df = ews.get_stress_periods(min_score=3)
    if not stress_df.empty:
        st.dataframe(stress_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Aucune période de stress détectée sur la période analysée.")

    # ── Statistiques globales ─────────────────────────
    st.markdown("### Résumé statistique")
    stats = ews.summary_stats()
    stats_df = pd.DataFrame(
        stats.items(), columns=["Indicateur", "Valeur"]
    )
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ── Export alertes ────────────────────────────────
    st.markdown("---")
    alerts_df = ews.get_alerts(min_score=3)
    if not alerts_df.empty:
        buffer_ews = io.BytesIO()
        alerts_df.to_csv(buffer_ews)
        st.download_button(
            label="⬇️ Exporter les alertes (CSV)",
            data=buffer_ews.getvalue(),
            file_name=f"alertes_ews_{date.today()}.csv",
            mime="text/csv",
        )


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    "<center><span style='color:#5B7FA6; font-size:0.8rem;'>"
    "EWS · CDG — Early Warning System | Module 1 : Construction de Portefeuille | "
    f"Données : Yahoo Finance | Mis à jour le {datetime.today().strftime('%d/%m/%Y')}"
    "</span></center>",
    unsafe_allow_html=True,
)
