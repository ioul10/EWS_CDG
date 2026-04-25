"""
app.py
──────
EWS-CDG — Early Warning System
Modules 1 & 2 : Construction de Portefeuille + Score EWS Global

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
import sys
import os

sys.path.append(os.path.dirname(__file__))

from data.masi20_tickers  import MASI20_UNIVERSE, get_display_label, get_sector, SECTOR_COLORS
from modules.data_loader  import fetch_prices, fetch_index, compute_simple_returns, validate_date_range
from modules.portfolio    import Portfolio, normalize_weights, equal_weights
from modules.markowitz    import MarkowitzOptimizer
from modules.ews_engine   import EWSEngine


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EWS · CDG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem; color: #1F4E79;
        letter-spacing: -0.5px; margin-bottom: 0;
    }
    .sub-title {
        font-size: 0.95rem; color: #5B7FA6;
        margin-top: 0.2rem; font-weight: 300; letter-spacing: 0.5px;
    }
    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem; color: #1F4E79;
        border-left: 4px solid #2E75B6;
        padding-left: 12px; margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fbff 0%, #edf4fd 100%);
        border: 1px solid #d0e4f7; border-radius: 12px;
        padding: 1rem 1.2rem; text-align: center;
    }
    .metric-label {
        font-size: 0.78rem; color: #5B7FA6; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.8px;
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem; color: #1F4E79; font-weight: 400;
    }
    .status-badge {
        display: inline-block; padding: 4px 12px;
        border-radius: 20px; font-size: 0.78rem; font-weight: 600;
    }
    .badge-success { background: #e8f5e9; color: #2e7d32; }
    .badge-info    { background: #e3f2fd; color: #1565c0; }
    .divider { border: none; border-top: 1px solid #e0ecf8; margin: 1.5rem 0; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1F4E79 0%, #0d2d4a 100%);
    }
    [data-testid="stSidebar"] * { color: #cce0f5 !important; }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff !important;
        font-family: 'DM Serif Display', serif;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-family: 'DM Sans', sans-serif;
        font-weight: 500; font-size: 0.95rem; transition: all 0.2s; width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #163a5c, #1F4E79);
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(31,78,121,0.3);
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────

_, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<div class="main-title">📊 Système d\'Alerte Précoce · CDG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">MODULES 1 & 2 — CONSTRUCTION DE PORTEFEUILLE + EARLY WARNING SYSTEM</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**Période d'analyse**")
    start_date = st.date_input("Date de début", value=date(2019, 1, 1), min_value=date(2010, 1, 1))
    end_date   = st.date_input("Date de fin",   value=date.today())

    st.markdown("---")
    st.markdown("**Valeur initiale du portefeuille**")
    initial_value = st.number_input(
        "Montant (MAD)", min_value=1_000_000, max_value=100_000_000_000,
        value=100_000_000, step=1_000_000, format="%d",
    )
    st.caption(f"≈ {initial_value/1_000_000:.0f} M MAD")

    st.markdown("---")
    st.markdown("**Mode d'allocation**")
    weight_mode = st.radio("", options=["Manuel", "Markowitz"], index=1, label_visibility="collapsed")

    if weight_mode == "Markowitz":
        objective = st.selectbox(
            "Objectif d'optimisation",
            options=["Maximiser le ratio de Sharpe", "Minimiser la variance", "Cibler un rendement"],
        )
        if objective == "Cibler un rendement":
            target_return = st.slider("Rendement cible (%)", 1, 25, 8) / 100
        max_weight = st.slider("Poids maximum par titre (%)", 10, 50, 40) / 100

    st.markdown("---")
    st.markdown("**Paramètres EWS**")
    risk_free   = st.slider("Taux sans risque (%)", 0, 10, 3) / 100
    vol_window  = st.slider("Fenêtre volatilité (jours)", 10, 60, 30)
    pca_window  = st.slider("Fenêtre PCA (jours)", 30, 120, 60)


# ─── Ticker Selection ─────────────────────────────────────────────────────────

st.markdown('<div class="section-header">① Sélection des titres</div>', unsafe_allow_html=True)

sectors = {}
for ticker, info in MASI20_UNIVERSE.items():
    s = info["sector"]
    sectors.setdefault(s, []).append(ticker)

selected_tickers = []
cols = st.columns(3)
for col_idx, (sector, tickers) in enumerate(sectors.items()):
    with cols[col_idx % 3]:
        st.markdown(f"**{sector}**")
        for ticker in tickers:
            label = f"{MASI20_UNIVERSE[ticker]['name']} `{ticker}`"
            if st.checkbox(label, value=True, key=f"chk_{ticker}"):
                selected_tickers.append(ticker)

if len(selected_tickers) < 2:
    st.warning("⚠️ Veuillez sélectionner au moins 2 titres.")
    st.stop()

st.markdown(f"<span class='status-badge badge-info'>✓ {len(selected_tickers)} titres sélectionnés</span>", unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Manual Weights ───────────────────────────────────────────────────────────

manual_weights = {}
if weight_mode == "Manuel":
    st.markdown('<div class="section-header">② Allocation manuelle des poids</div>', unsafe_allow_html=True)

    equal_w   = 100 / len(selected_tickers)
    total_pct = 0
    cols_w    = st.columns(min(4, len(selected_tickers)))

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

    if abs(total_pct - 100) > 0.1:
        st.error(f"❌ Somme des poids : {total_pct:.1f}% — doit être 100%.")
        st.stop()
    else:
        st.success(f"✅ Somme des poids : {total_pct:.1f}%")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Build Button ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">③ Construction & Analyse</div>', unsafe_allow_html=True)

build_col, _ = st.columns([2, 5])
with build_col:
    build = st.button("🚀 Construire le portefeuille & lancer l'EWS", use_container_width=True)

if not build:
    st.info("👆 Configurez vos paramètres, sélectionnez vos titres, puis cliquez sur le bouton.")
    st.stop()


# ─── Data Loading ─────────────────────────────────────────────────────────────

with st.spinner("📡 Chargement des données..."):
    try:
        validate_date_range(str(start_date), str(end_date))
        prices, valid_tickers, sources = fetch_prices(
            selected_tickers, start=str(start_date), end=str(end_date),
        )
        masi = fetch_index(start=str(start_date), end=str(end_date))
    except ValueError as e:
        st.error(f"❌ {e}")
        st.markdown("""
        **Comment résoudre :**
        1. Allez sur [fr.investing.com](https://fr.investing.com)
        2. Cherchez le titre → **Données Historiques** → **Télécharger**
        3. Renommez : `ATW.csv`, `IAM.csv`, etc.
        4. Placez dans `data/prices/`
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur inattendue : {e}")
        st.stop()

if len(valid_tickers) < 2:
    st.error("❌ Moins de 2 titres chargés.")
    st.stop()

csv_tickers    = [t for t, s in sources.items() if s == "csv"]
yahoo_tickers  = [t for t, s in sources.items() if s == "yahoo"]
failed_tickers = [t for t, s in sources.items() if s == "failed"]

if csv_tickers:   st.success(f"✅ {len(csv_tickers)} titre(s) — CSV local")
if yahoo_tickers: st.info(f"📡 {len(yahoo_tickers)} titre(s) — Yahoo Finance")
if failed_tickers:st.warning(f"⚠️ Non disponibles : {', '.join(failed_tickers)}")


# ─── Portfolio Construction ───────────────────────────────────────────────────

returns_df = compute_simple_returns(prices)

with st.spinner("⚙️ Construction du portefeuille..."):
    if weight_mode == "Markowitz":
        optimizer = MarkowitzOptimizer(returns=returns_df, risk_free_rate=risk_free, max_weight=max_weight)
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

    portfolio = Portfolio(
        prices=prices, weights=weights_dict,
        initial_value=initial_value, name="Portefeuille MASI20 — CDG",
    )

# ─── EWS Engine ───────────────────────────────────────────────────────────────

with st.spinner("🧠 Calcul du score EWS (Statistique + ML)... Cela peut prendre 30 secondes."):
    ews     = EWSEngine(portfolio=portfolio, vol_window=vol_window, pca_window=pca_window)
    current = ews.get_current_status()

st.success("✅ Portefeuille construit — Score EWS calculé.")
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── KPI Summary ──────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">④ Résultats</div>', unsafe_allow_html=True)

summary  = portfolio.get_summary()
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
        </div>""", unsafe_allow_html=True)

st.markdown("")
val_cols = st.columns(2)
with val_cols[0]:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">💰 Valeur initiale</div>
        <div class="metric-value">{summary['Valeur initiale (MAD)']}</div>
    </div>""", unsafe_allow_html=True)
with val_cols[1]:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">💼 Valeur finale</div>
        <div class="metric-value">{summary['Valeur finale (MAD)']}</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Évolution", "🥧 Allocation",
    "📊 Frontière efficiente", "📋 Données", "🚨 Alertes EWS"
])

# ── Tab 1 — Evolution ─────────────────────────────────────────────────────────
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio.total_value.index, y=portfolio.total_value.values,
        mode="lines", name=portfolio.name,
        line=dict(color="#1F4E79", width=2.5),
        fill="tozeroy", fillcolor="rgba(31,78,121,0.08)",
    ))
    masi_aligned    = masi.reindex(portfolio.total_value.index).ffill()
    masi_normalized = masi_aligned / masi_aligned.iloc[0] * initial_value
    fig.add_trace(go.Scatter(
        x=masi_normalized.index, y=masi_normalized.values,
        mode="lines", name="MASI (normalisé)",
        line=dict(color="#ED7D31", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title="Évolution de la valeur du portefeuille vs MASI",
        xaxis_title="Date", yaxis_title="Valeur (MAD)",
        template="plotly_white", height=450,
        font=dict(family="DM Sans", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2 — Allocation ────────────────────────────────────────────────────────
with tab2:
    weights_series = pd.Series(weights_dict)
    labels = [MASI20_UNIVERSE[t]["name"] for t in weights_series.index]
    colors = [SECTOR_COLORS.get(get_sector(t), "#999999") for t in weights_series.index]

    fig_pie = go.Figure(go.Pie(
        labels=labels, values=weights_series.values, hole=0.45,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+percent", textfont=dict(family="DM Sans", size=12),
    ))
    fig_pie.update_layout(
        title=f"Allocation — {weight_mode}", template="plotly_white",
        height=500, font=dict(family="DM Sans"), showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    w_df = pd.DataFrame({
        "Titre":               labels,
        "Ticker":              list(weights_series.index),
        "Secteur":             [get_sector(t) for t in weights_series.index],
        "Poids (%)":           (weights_series.values * 100).round(2),
        "Capital alloué (MAD)":(weights_series.values * initial_value).round(0).astype(int),
    }).sort_values("Poids (%)", ascending=False).reset_index(drop=True)
    st.dataframe(w_df, use_container_width=True, hide_index=True)

# ── Tab 3 — Efficient Frontier ────────────────────────────────────────────────
with tab3:
    if frontier_df is not None and mc_df is not None:
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=mc_df["Volatilité"] * 100, y=mc_df["Rendement"] * 100,
            mode="markers",
            marker=dict(size=3, color=mc_df["Sharpe"], colorscale="Blues",
                        showscale=True, colorbar=dict(title="Sharpe"), opacity=0.5),
            name="Portefeuilles simulés",
        ))
        fig_ef.add_trace(go.Scatter(
            x=frontier_df["Volatilité"] * 100, y=frontier_df["Rendement"] * 100,
            mode="lines", line=dict(color="#1F4E79", width=3),
            name="Frontière efficiente",
        ))
        fig_ef.add_trace(go.Scatter(
            x=[opt_result["volatility"] * 100], y=[opt_result["expected_return"] * 100],
            mode="markers", marker=dict(color="#ED7D31", size=14, symbol="star"),
            name=f"Optimal ({opt_result['label']})",
        ))
        fig_ef.update_layout(
            title="Frontière Efficiente de Markowitz — MASI20",
            xaxis_title="Volatilité annualisée (%)", yaxis_title="Rendement annualisé (%)",
            template="plotly_white", height=500,
            font=dict(family="DM Sans", size=13),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ef, use_container_width=True)
        if opt_result:
            oc = st.columns(3)
            oc[0].metric("Rendement espéré", f"{opt_result['expected_return']*100:.2f}%")
            oc[1].metric("Volatilité",        f"{opt_result['volatility']*100:.2f}%")
            oc[2].metric("Ratio de Sharpe",   f"{opt_result['sharpe_ratio']:.2f}")
    else:
        st.info("Frontière efficiente disponible uniquement en mode Markowitz.")

# ── Tab 4 — Raw Data ──────────────────────────────────────────────────────────
with tab4:
    st.markdown("**Cours de clôture ajustés (30 derniers jours)**")
    st.dataframe(prices.tail(30).style.format("{:.2f}"), use_container_width=True)
    st.markdown("**Rendements journaliers**")
    st.dataframe(returns_df.tail(30).style.format("{:.4%}"), use_container_width=True)
    st.markdown("---")
    buf = io.BytesIO()
    portfolio.total_value.reset_index().to_csv(buf, index=False)
    st.download_button("⬇️ Exporter la valeur du portefeuille (CSV)",
                       buf.getvalue(), f"portefeuille_{date.today()}.csv", "text/csv")

# ── Tab 5 — EWS Alertes ───────────────────────────────────────────────────────
with tab5:

    # Score global du jour
    score   = current["score_global"]
    couleur = "#2e7d32" if score < 4 else "#f57f17" if score < 7 else "#c62828"

    st.markdown("### Score EWS Global — Aujourd'hui")
    st.markdown(f"""
    <div style="background:{couleur}22; border-left:6px solid {couleur};
                padding:1.2rem 1.5rem; border-radius:10px; margin-bottom:1rem;">
        <div style="font-size:1.8rem; font-weight:700; color:{couleur};">
            {current['niveau_global']}
        </div>
        <div style="color:#555; margin-top:0.3rem;">
            Score global : <strong>{score} / 10</strong> — {current['date']}
        </div>
        <div style="margin-top:0.5rem; font-size:0.9rem; color:{couleur}; font-weight:600;">
            ⚡ Action recommandée : {current['action']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if current["signal_hedge"]:
        st.error("🚨 Signal de hedging déclenché — Module 3 activé")

    # Décomposition du score
    st.markdown("### Décomposition du score")
    col_s, col_m, col_g = st.columns(3)

    def big_card(col, title, score_val, max_val, color, subtitle=""):
        pct = (score_val / max_val * 100) if max_val > 0 else 0
        col.markdown(f"""
        <div style="border:2px solid {color}55; border-radius:12px;
                    padding:1rem; text-align:center; background:{color}11;">
            <div style="font-size:0.75rem; color:#666; font-weight:600;
                        text-transform:uppercase; letter-spacing:0.8px;">{title}</div>
            <div style="font-size:2rem; font-weight:700; color:{color}; margin:0.4rem 0;">
                {score_val} <span style="font-size:1rem; color:#999;">/ {max_val}</span>
            </div>
            <div style="height:6px; background:#eee; border-radius:3px;">
                <div style="width:{pct}%; height:100%; background:{color}; border-radius:3px;"></div>
            </div>
            <div style="font-size:0.75rem; color:#888; margin-top:0.4rem;">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

    g_color = "#c62828" if score >= 7 else "#f57f17" if score >= 4 else "#2e7d32"
    big_card(col_s, "Couche Statistique", current["score_stat"],   8,  "#1F4E79", "Vol + Z-score + Drawdown + VaR")
    big_card(col_m, "Couche ML",          current["score_ml"],     6,  "#2E75B6", "Isolation Forest + PCA + Régimes")
    big_card(col_g, "Score Global EWS",   current["score_global"], 10, g_color,   f"Régime : {current['regime']}")

    st.markdown("")

    # Indicateurs du jour
    st.markdown("### Indicateurs du jour")
    c1, c2, c3, c4, c5 = st.columns(5)

    def small_card(col, label, value, note=""):
        col.markdown(f"""
        <div style="border:1px solid #d0e4f7; border-radius:8px;
                    padding:0.7rem; text-align:center; background:#f8fbff;">
            <div style="font-size:0.7rem; color:#5B7FA6; font-weight:600;
                        text-transform:uppercase;">{label}</div>
            <div style="font-size:1.2rem; font-weight:700; color:#1F4E79;">{value}</div>
            <div style="font-size:0.7rem; color:#999;">{note}</div>
        </div>
        """, unsafe_allow_html=True)

    small_card(c1, "Volatilité 30j", current["volatilite"])
    small_card(c2, "Z-Score",        current["z_score"])
    small_card(c3, "Drawdown",       current["drawdown"])
    small_card(c4, "VaR 99%",        current["var_99"])
    small_card(c5, "PC1 Variance",   current["pc1_variance"],
               "⚠️ Stress systémique" if current.get("pca_stress") else "Normal")

    st.markdown("")

    # Évolution du score EWS
    st.markdown("### Évolution du Score EWS Global")
    ews_plot = ews.results[["Score_Global", "Score_Stat", "Score_ML"]].dropna()

    fig_ews = go.Figure()
    fig_ews.add_hrect(y0=0, y1=4,  fillcolor="#2e7d32", opacity=0.05, line_width=0)
    fig_ews.add_hrect(y0=4, y1=7,  fillcolor="#f57f17", opacity=0.05, line_width=0)
    fig_ews.add_hrect(y0=7, y1=10, fillcolor="#c62828", opacity=0.05, line_width=0)

    fig_ews.add_trace(go.Scatter(
        x=ews_plot.index, y=ews_plot["Score_Global"],
        mode="lines", name="Score Global EWS",
        line=dict(color="#1F4E79", width=2.5),
        fill="tozeroy", fillcolor="rgba(31,78,121,0.08)",
    ))
    fig_ews.add_trace(go.Scatter(
        x=ews_plot.index, y=ews_plot["Score_Stat"] / 8 * 10,
        mode="lines", name="Score Stat (normalisé)",
        line=dict(color="#2E75B6", width=1, dash="dot"), opacity=0.6,
    ))
    fig_ews.add_trace(go.Scatter(
        x=ews_plot.index, y=ews_plot["Score_ML"] / 6 * 10,
        mode="lines", name="Score ML (normalisé)",
        line=dict(color="#ED7D31", width=1, dash="dot"), opacity=0.6,
    ))
    fig_ews.add_hline(y=4, line=dict(color="#f57f17", dash="dash", width=1.5),
                      annotation_text="Vigilance (4)", annotation_position="right")
    fig_ews.add_hline(y=7, line=dict(color="#c62828", dash="dash", width=1.5),
                      annotation_text="Critique (7)", annotation_position="right")

    fig_ews.update_layout(
        xaxis_title="Date", yaxis_title="Score EWS (/10)",
        yaxis=dict(range=[0, 10.5]), template="plotly_white", height=420,
        font=dict(family="DM Sans", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_ews, use_container_width=True)

    # Périodes de stress
    st.markdown("### Périodes de stress détectées")
    stress_df = ews.get_stress_periods()
    if not stress_df.empty:
        st.dataframe(stress_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Aucune période critique détectée sur la période analysée.")

    # Signaux hedge
    hedge_df = ews.get_hedge_signals()
    if not hedge_df.empty:
        st.markdown(f"### 🚨 Signaux de hedging — {len(hedge_df)} jours")
        st.dataframe(hedge_df.head(20), use_container_width=True)

    # Résumé statistique
    st.markdown("### Résumé statistique")
    stats_ews = ews.summary_stats()
    st.dataframe(
        pd.DataFrame(stats_ews.items(), columns=["Indicateur", "Valeur"]),
        use_container_width=True, hide_index=True,
    )

    # Export
    st.markdown("---")
    buf_ews = io.BytesIO()
    ews.results.to_csv(buf_ews)
    st.download_button(
        label="⬇️ Exporter les résultats EWS complets (CSV)",
        data=buf_ews.getvalue(),
        file_name=f"ews_results_{date.today()}.csv",
        mime="text/csv",
    )


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    f"<center><span style='color:#5B7FA6; font-size:0.8rem;'>"
    f"EWS · CDG — Early Warning System | Modules 1 & 2 | "
    f"Mis à jour le {datetime.today().strftime('%d/%m/%Y')}"
    f"</span></center>",
    unsafe_allow_html=True,
)
