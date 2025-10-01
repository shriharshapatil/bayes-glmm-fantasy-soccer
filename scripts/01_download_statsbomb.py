# scripts/01_download_statsbomb.py
from statsbombpy import sb
import os
import json
import pandas as pd
from tqdm import tqdm

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

def list_competitions():
    comps = sb.competitions()
    print(comps.head())
    return comps

def download_matches(competition_id: int, season_id: int):
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    matches.to_csv(os.path.join(RAW_DIR, f"matches_{competition_id}_{season_id}.csv"), index=False)
    return matches

def download_all_events_for_matches(matches_df: pd.DataFrame):
    for mid in tqdm(matches_df['match_id'].unique(), desc="matches"):
        try:
            events = sb.events(match_id=mid)
            events.to_json(os.path.join(RAW_DIR, f"events_{mid}.json"), orient="records")
            # lineups are often helpful (starting XI + subs)
            lineups = sb.lineups(match_id=mid)
            # save lineups as one JSON per match
            with open(os.path.join(RAW_DIR, f"lineups_{mid}.json"), "w", encoding="utf-8") as f:
                json.dump(lineups, f, default=str)
        except Exception as e:
            print(f"failed {mid}: {e}")

if __name__ == "__main__":
    # EXAMPLE: find competition_id and season_id from sb.competitions()
    # comps = list_competitions()   # uncomment to inspect
    # then set the IDs below
    competition_id = 43   # replace with the competition you want
    season_id = 106         # replace with the season id
    matches = download_matches(competition_id, season_id)
    download_all_events_for_matches(matches)
