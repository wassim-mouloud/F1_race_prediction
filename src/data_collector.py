"""
Data Collection Module using FastF1 API
"""

import fastf1
import pandas as pd
from typing import Optional
from pathlib import Path


class F1DataCollector:
    """
    Collects F1 data using the FastF1 API.
    """

    def __init__(self, cache_dir: str = './cache'):
        """
        Initialize the data collector.

        Args:
            cache_dir: Directory to cache FastF1 data (not used - caching disabled)
        """
        self.cache_dir = Path(cache_dir)
        # Caching disabled - FastF1 will fetch fresh data every time
        # self.cache_dir.mkdir(exist_ok=True)
        # fastf1.Cache.enable_cache(str(self.cache_dir))

    def get_season_schedule(self, year: int) -> pd.DataFrame:
        """
        Get the race schedule for a specific season.

        Args:
            year: Season year

        Returns:
            DataFrame with race schedule
        """
        schedule = fastf1.get_event_schedule(year)
        return schedule

    def get_session_data(self, year: int, race: str, session_type: str = 'R') -> fastf1.core.Session: # type: ignore
        """
        Get data for a specific session.

        Args:
            year: Season year
            race: Race name or round number
            session_type: 'FP1', 'FP2', 'FP3', 'Q', 'R' (Race)

        Returns:
            Session object with loaded data
        """
        session = fastf1.get_session(year, race, session_type)
        session.load()
        return session

    def get_driver_lap_times(self, session: fastf1.core.Session, driver: str) -> pd.DataFrame: # type: ignore
        """
        Get lap times for a specific driver in a session.

        Args:
            session: FastF1 session object
            driver: Driver abbreviation (e.g., 'VER', 'HAM')

        Returns:
            DataFrame with lap times
        """
        laps = session.laps.pick_driver(driver).pick_fastest()
        return laps

    def get_race_results(self, session: fastf1.core.Session) -> pd.DataFrame: # type: ignore
        """
        Get race results.

        Args:
            session: FastF1 session object (must be race session)

        Returns:
            DataFrame with race results
        """
        return session.results

    def collect_historical_data(
        self,
        start_year: int,
        end_year: int,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Collect historical race data for multiple seasons.

        Args:
            start_year: Starting season
            end_year: Ending season
            save_path: Optional path to save the collected data

        Returns:
            DataFrame with historical race data
        """
        all_data = []

        for year in range(start_year, end_year + 1):
            try:
                schedule = self.get_season_schedule(year)

                for _, race_event in schedule.iterrows():
                    if pd.notna(race_event['EventDate']):
                        try:
                            session = self.get_session_data(year, race_event['RoundNumber'], 'R')
                            results = self.get_race_results(session)

                            # Add metadata
                            results['Year'] = year
                            results['RaceName'] = race_event['EventName']
                            results['Round'] = race_event['RoundNumber']

                            all_data.append(results)
                        except Exception as e:
                            print(f"Error collecting data for {year} - {race_event['EventName']}: {e}")
            except Exception as e:
                print(f"Error processing year {year}: {e}")

        if all_data:
            df = pd.concat(all_data, ignore_index=True)

            if save_path:
                df.to_csv(save_path, index=False)
                print(f"Data saved to {save_path}")

            return df

        return pd.DataFrame()
