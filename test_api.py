"""
Script de test simple pour vérifier la connexion à l'API FastF1
"""

import pandas as pd
from requests import session
from src.data_collector import F1DataCollector

def test_api_connection():
    """Test simple pour vérifier si on reçoit la data de l'API"""

    print("🔄 Initialisation du collector...")
    collector = F1DataCollector()

    print("\n🏁 Test 2: Récupération des données d'une course...")
    try:
        session = collector.get_session_data(2025, "3", 'Q')
        print(f"✅ Succès! Session chargée: {session.event['EventName']}")
        print(f"   Date: {session.event['EventDate']}")

        print("\n=== Fastest Lap per Driver ===")
        # Get fastest lap for each driver
        fastest_laps = session.laps.loc[session.laps.groupby('Driver')['LapTime'].idxmin()]
        fastest_laps = fastest_laps[['Driver', 'LapTime']].sort_values('LapTime').reset_index(drop=True)

        # Calculate time difference to next driver
        fastest_laps['Gap'] = fastest_laps['LapTime'].diff().shift(-1)

        for _, row in fastest_laps.iterrows():
            if pd.notna(row['Gap']):
                gap_seconds = row['Gap'].total_seconds()
                print(f"{row['Driver']}: {row['LapTime']} (Gap to next: +{gap_seconds:.3f}s)")
            else:
                print(f"{row['Driver']}: {row['LapTime']}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return

    


if __name__ == "__main__":
    test_api_connection()
