"""
Script de test simple pour vérifier la connexion à l'API FastF1
"""


from src.data_collector import F1DataCollector

def test_api_connection():
    """Test simple pour vérifier si on reçoit la data de l'API"""

    print("🔄 Initialisation du collector...")
    collector = F1DataCollector()

    # Test with 2023 season which has complete data
    print("\n🏁 Test 1: Monaco GP 2023...")
    try:
        session = collector.get_session_data(2023, "Monaco", 'R')
        print(f"✅ Succès! Session chargée: {session.event['EventName']}")
        print(f"   Date: {session.event['EventDate']}")
        print(f"   Round: {session.event['RoundNumber']}")
        print(f"   Location: {session.event['Location']}")
        print(f"   Winner: {session.results.iloc[0]['Abbreviation']}")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return

    print("\n🏁 Test 2: Monza GP 2023...")
    try:
        session = collector.get_session_data(2023, "Monza", 'R')
        print(f"✅ Succès! Session chargée: {session.event['EventName']}")
        print(f"   Date: {session.event['EventDate']}")
        print(f"   Round: {session.event['RoundNumber']}")
        print(f"   Location: {session.event['Location']}")

        # Show all columns
        print(f"\n📊 All columns: {list(session.results.columns)}")

        # Check which columns have data (non-NaN)
        print(f"\n📈 Columns with data:")
        for col in session.results.columns:
            non_null = session.results[col].notna().sum()
            if non_null > 0:
                print(f"   {col}: {non_null}/{len(session.results)} rows")

        # Show only relevant race data (use ClassifiedPosition, not Position!)
        race_cols = ['ClassifiedPosition', 'DriverNumber', 'Abbreviation', 'TeamName', 'Status']
        print(f"\n🏎️  Race Results (Top 10):\n{session.results[race_cols].head(10)}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return

    


if __name__ == "__main__":
    test_api_connection()
