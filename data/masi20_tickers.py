"""
MASI 20 — Universe of Stocks
Casablanca Stock Exchange (BVC)
Tickers formatted for Yahoo Finance (.CS suffix)
"""

MASI20_UNIVERSE = {
    "ATW.CS":    {"name": "Attijariwafa Bank",      "sector": "Banques"},
    "BCP.CS":    {"name": "Banque Centrale Populaire","sector": "Banques"},
    "CIH.CS":    {"name": "CIH Bank",               "sector": "Banques"},
    "BMAO.CS":   {"name": "Bank of Africa",          "sector": "Banques"},
    "IAM.CS":    {"name": "Maroc Telecom",           "sector": "Télécommunications"},
    "LHM.CS":    {"name": "Lafarge Holcim Maroc",   "sector": "Ciment & Matériaux"},
    "CSR.CS":    {"name": "Cosumar",                 "sector": "Agroalimentaire"},
    "HPS.CS":    {"name": "HPS",                     "sector": "Technologie"},
    "MARSA.CS":  {"name": "Marsa Maroc",             "sector": "Transport & Logistique"},
    "SID.CS":    {"name": "Sonasid",                 "sector": "Sidérurgie"},
    "ADH.CS":    {"name": "Addoha",                  "sector": "Immobilier"},
    "ADI.CS":    {"name": "Alliances",               "sector": "Immobilier"},
    "AUTO.CS":   {"name": "Auto Hall",               "sector": "Distribution"},
    "DHO.CS":    {"name": "Douja Prom",              "sector": "Immobilier"},
    "LBV.CS":    {"name": "Label Vie",               "sector": "Distribution"},
    "MNG.CS":    {"name": "Managem",                 "sector": "Mines"},
    "MUT.CS":    {"name": "Mutandis",                "sector": "Agroalimentaire"},
    "REB.CS":    {"name": "Rebab Company",           "sector": "Immobilier"},
    "RDS.CS":    {"name": "Résidences Dar Saada",    "sector": "Immobilier"},
    "STROC.CS":  {"name": "Stroc Industrie",         "sector": "Industrie"},
}

# Reference index
MASI_INDEX = "^MASI"

# Sector color mapping (for charts)
SECTOR_COLORS = {
    "Banques":                  "#1F4E79",
    "Télécommunications":       "#2E75B6",
    "Ciment & Matériaux":       "#70AD47",
    "Agroalimentaire":          "#ED7D31",
    "Technologie":              "#4472C4",
    "Transport & Logistique":   "#5B9BD5",
    "Sidérurgie":               "#636363",
    "Immobilier":               "#FFC000",
    "Distribution":             "#FF0000",
    "Mines":                    "#7030A0",
    "Industrie":                "#C55A11",
}

def get_ticker_list():
    """Returns list of all tickers."""
    return list(MASI20_UNIVERSE.keys())

def get_name(ticker: str) -> str:
    """Returns company name for a given ticker."""
    return MASI20_UNIVERSE.get(ticker, {}).get("name", ticker)

def get_sector(ticker: str) -> str:
    """Returns sector for a given ticker."""
    return MASI20_UNIVERSE.get(ticker, {}).get("sector", "Inconnu")

def get_display_label(ticker: str) -> str:
    """Returns display label: 'Name (TICKER)' format."""
    name = get_name(ticker)
    return f"{name} ({ticker})"
