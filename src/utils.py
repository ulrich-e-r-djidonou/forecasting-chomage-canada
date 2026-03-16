"""
Fonctions utilitaires pour le projet Forecasting du chomage canadien.
Source des donnees : Statistique Canada, Table 14-10-0287-03
Licence : Licence du gouvernement ouvert - Canada
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
from pathlib import Path

# ============================================================
# Chemins du projet
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_DIR / "data" / "raw"
DATA_PROCESSED = PROJECT_DIR / "data" / "processed"
OUTPUTS_FIGURES = PROJECT_DIR / "outputs" / "figures"

# ============================================================
# Constantes
# ============================================================
PROVINCES = [
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Nova Scotia", "Ontario",
    "Prince Edward Island", "Quebec", "Saskatchewan"
]

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

# Table 14-10-0287-03 : Caractéristiques de la population active, données mensuelles, désaisonnalisées
STATCAN_TABLE_ID = "14-10-0287-03"
STATCAN_CSV_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/14100287-eng.zip"


def download_statcan_table(url: str = STATCAN_CSV_URL,
                           save_dir: str | Path = DATA_RAW,
                           timeout: int = 120) -> pd.DataFrame:
    """
    Telecharge et extrait la table StatCan depuis le fichier ZIP.
    Retourne le DataFrame brut.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "14100287.csv"

    # Si deja telecharge, lire le fichier local
    if csv_path.exists():
        print(f"Fichier deja present : {csv_path}")
        print("Chargement depuis le cache local...")
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"Shape : {df.shape}")
        return df

    print(f"Telechargement depuis : {url}")
    print("Cela peut prendre quelques minutes...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # Retry avec backoff en cas d'echec reseau
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Tentative {attempt}/{max_retries}...")
            r = requests.get(url, timeout=timeout, headers=headers, stream=True)
            r.raise_for_status()
            content = r.content
            break
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Impossible de telecharger apres {max_retries} tentatives.\n"
                    f"Verifiez votre connexion ou telechargez manuellement :\n"
                    f"  {url}\n"
                    f"Placez le ZIP dans : {save_dir}"
                ) from e
            import time
            wait = 5 * attempt
            print(f"  Echec ({e.__class__.__name__}), nouvel essai dans {wait}s...")
            time.sleep(wait)

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv') and 'MetaData' not in f]
        print(f"Fichiers dans le ZIP : {z.namelist()}")
        csv_name = csv_files[0]
        z.extractall(save_dir)

        # Renommer pour coherence
        extracted = save_dir / csv_name
        if extracted != csv_path:
            extracted.rename(csv_path)

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Telechargement reussi! Shape : {df.shape}")
    return df


def filter_unemployment_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre le DataFrame brut StatCan pour ne garder que :
    - Le taux de chomage (Unemployment rate)
    - Les 10 provinces + Canada
    - Les deux sexes
    - 15 ans et plus
    Retourne un DataFrame avec colonnes : date, geo, unemployment_rate
    """
    # Identifier les colonnes pertinentes
    # La table StatCan utilise des noms de colonnes standardises
    col_geo = "GEO"
    col_date = "REF_DATE"
    col_indicator = "Labour force characteristics"
    col_value = "VALUE"
    col_gender = "Gender"
    col_age = "Age group"
    col_data_type = "Data type"
    col_stats = "Statistics"

    # Filtrer
    mask = (
        (df[col_indicator] == "Unemployment rate") &
        (df[col_gender] == "Total - Gender") &
        (df[col_age] == "15 years and over") &
        (df[col_data_type] == "Seasonally adjusted") &
        (df[col_stats] == "Estimate") &
        (df[col_geo].isin(PROVINCES + ["Canada"]))
    )

    result = df.loc[mask, [col_date, col_geo, col_value]].copy()
    result.columns = ["date", "geo", "unemployment_rate"]

    # Conversion des types
    result["date"] = pd.to_datetime(result["date"])
    result["unemployment_rate"] = pd.to_numeric(result["unemployment_rate"], errors="coerce")
    result = result.sort_values(["geo", "date"]).reset_index(drop=True)

    print(f"Apres filtrage : {result.shape[0]} observations")
    print(f"Periode : {result['date'].min()} a {result['date'].max()}")
    print(f"Geographies : {result['geo'].nunique()} ({', '.join(sorted(result['geo'].unique()))})")

    return result


def prepare_prophet_df(df: pd.DataFrame, geo: str = "Canada") -> pd.DataFrame:
    """
    Prepare un DataFrame au format Prophet (colonnes ds, y) pour une geographie donnee.
    """
    subset = df[df["geo"] == geo][["date", "unemployment_rate"]].copy()
    subset.columns = ["ds", "y"]
    subset = subset.dropna().sort_values("ds").reset_index(drop=True)
    return subset


def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcule les metriques de prevision : MAE, RMSE, MAPE.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE": round(mape, 2)}
