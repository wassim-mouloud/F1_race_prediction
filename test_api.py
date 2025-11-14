"""
Script de test simple pour vérifier la connexion à l'API FastF1
"""

from src.data_collector import F1DataCollector

def test_api_connection():
    """Test simple pour vérifier si on reçoit la data de l'API"""

    print("🔄 Initialisation du collector...")
    collector = F1DataCollector()

    print("\n📅 Test 1: Récupération du calendrier 2024...")
    try:
        schedule = collector.get_season_schedule(2018)
        print(f"✅ Succès! Nombre de courses: {len(schedule)}")
        print("\nPremières courses de la saison:")
        print(schedule[['RoundNumber', 'EventName', 'EventDate']].head())
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return

    print("\n🏁 Test 2: Récupération des données d'une course...")
    try:
        # Test avec la première course de 2018 (Australie)
        session = collector.get_session_data(2018, "1", 'Q')
        print(f"✅ Succès! Session chargée: {session.event['EventName']}")
        print(f"   Date: {session.event['EventDate']}")
        print(f"   Nombre de tours enregistrés: {len(session.laps)}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return

    # print("\n🏎️  Test 3: Récupération des résultats de la course...")
    # try:
    #     results = collector.get_race_results(session)
    #     print(f"✅ Succès! Nombre de pilotes: {len(results)}")
    #     print("\nTop 5:")
    #     print(results[['Position', 'Abbreviation', 'TeamName', 'Points']].head())
    # except Exception as e:
    #     print(f"❌ Erreur: {e}")
    #     return
    print("\n Hamilton lap time : ", collector.get_driver_lap_times(session, '44'))

    print("\n✨ Tous les tests ont réussi! L'API fonctionne correctement.")
    


if __name__ == "__main__":
    test_api_connection()
