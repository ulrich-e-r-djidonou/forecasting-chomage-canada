r"""
Script de mise à jour automatique des données StatCan.
Télécharge la dernière version de la table 14-10-0287-03 et met à jour
le fichier processed.

Usage :
    python scripts/update_data.py

La mise à jour tourne aussi toute seule via GitHub Actions
(.github/workflows/update_monthly.yml), le 5 de chaque mois.

Pour automatiser en local (Windows Task Scheduler) :
    - Programme : <chemin vers python.exe>
    - Arguments : <racine du dépôt>\scripts\update_data.py
    - Fréquence : mensuelle (le 15 de chaque mois, après la publication EPA)
"""

import sys
from pathlib import Path

# Ajouter le dossier src au path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from utils import download_statcan_table, filter_unemployment_rate, DATA_RAW, DATA_PROCESSED
from datetime import datetime


def main():
    print(f"=== Mise à jour des données — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")

    # Supprimer le cache pour forcer le re-téléchargement
    csv_cache = DATA_RAW / "14100287.csv"
    if csv_cache.exists():
        print(f"Suppression du cache : {csv_cache}")
        csv_cache.unlink()

    # Télécharger
    print("\n1. Téléchargement de la table StatCan...")
    df_raw = download_statcan_table()

    # Filtrer
    print("\n2. Filtrage du taux de chômage...")
    df = filter_unemployment_rate(df_raw)

    # Sauvegarder
    print("\n3. Sauvegarde...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "chomage_mensuel.csv"
    df.to_csv(output_path, index=False)
    print(f"   Fichier mis à jour : {output_path}")
    print(f"   Dernière observation : {df['date'].max()}")
    print(f"   Nombre d'observations : {df.shape[0]}")

    print("\n=== Mise à jour terminée ===")
    print("Vous pouvez maintenant relancer le dashboard Streamlit.")


if __name__ == "__main__":
    main()
