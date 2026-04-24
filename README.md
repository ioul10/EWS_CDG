# 📊 EWS-CDG — Early Warning System

> Système d'Alerte Précoce et de Couverture Dynamique du Risque de Marché  
> Projet de Fin d'Études — Ingénierie Financière

---

## 🏗️ Architecture du projet

```
EWS-CDG/
│
├── app.py                        # Interface Streamlit principale
│
├── modules/
│   ├── data_loader.py            # Récupération données Yahoo Finance
│   ├── portfolio.py              # Construction & valorisation portefeuille
│   ├── markowitz.py              # Optimisation Markowitz (MPT)
│   ├── ews.py                    # Module 2 — Early Warning System (à venir)
│   └── hedging.py                # Module 3 — Hedging Futures MASI20 (à venir)
│
├── data/
│   └── masi20_tickers.py         # Univers MASI20 + métadonnées
│
├── requirements.txt
└── README.md
```

---

## 🚀 Modules

| Module | Statut | Description |
|--------|--------|-------------|
| **Module 1** — Construction de Portefeuille | ✅ Disponible | Sélection titres MASI20, allocation manuelle ou Markowitz, frontière efficiente |
| **Module 2** — Early Warning System | 🔄 En développement | VaR, Z-score, Drawdown, Isolation Forest, PCA, Régimes de marché |
| **Module 3** — Hedging Futures MASI20 | 🔄 En développement | Beta, pricer futures, stratégie de couverture dynamique |

---

## ⚙️ Installation

```bash
# Cloner le repository
git clone https://github.com/[username]/EWS-CDG.git
cd EWS-CDG

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

---

## 📦 Données

- **Source** : Yahoo Finance (tickers `.CS` — Casablanca Stock Exchange)
- **Univers** : 20 titres du MASI20
- **Période** : 2019 → Aujourd'hui
- **Fréquence** : Journalière (jours ouvrés)

---

## 🔬 Méthodologie

### Module 1 — Théorie Moderne du Portefeuille (Markowitz)
- Minimisation de la variance
- Maximisation du ratio de Sharpe
- Portefeuille à rendement cible
- Frontière efficiente (analytique + Monte Carlo)

### Module 2 — Early Warning System *(à venir)*
- **Couche statistique** : VaR historique rolling, Z-score, Drawdown, Volatilité rolling
- **Couche ML** : Isolation Forest, PCA, K-Means (régimes de marché)
- Score composite d'alerte (0 → 10)

### Module 3 — Hedging dynamique *(à venir)*
- Calcul du beta portefeuille vs MASI20
- Pricer contrats futures MASI20
- Stratégie de couverture déclenchée par les signaux EWS

---

## 📄 Licence

Usage académique — Projet de Fin d'Études  
© 2025 [Votre Nom] — [Établissement]
