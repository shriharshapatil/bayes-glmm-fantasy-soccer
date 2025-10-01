# scripts/02_feature_engineer_wc.py
"""
Feature engineer for StatsBomb World Cup (competition_id=43, season_id=106).
Robust version: handles events list + lineup dict/list shapes observed in your dataset.
Produces: data/processed/player_match_wc.csv
"""
import os
import glob
import json
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

RAW_DIR = "data/raw"
MATCHES_CSV = os.path.join(RAW_DIR, "matches_43_106.csv")  # adjust if name differs
OUT_CSV = "data/processed/player_match_wc.csv"
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# Simple, adjustable scoring rules (example)
SCORING = {
    "goal": {"F": 6, "M": 5, "D": 6, "GK": 6},  # if position missing, default to M value
    "assist": 3,
    "shot_on_target": 1,
    "key_pass": 1,
    "tackle": 1,
    "yellow_card": -1,
    "red_card": -3,
    "clean_sheet_DEF": 4,
    "clean_sheet_GK": 6
}

def safeget(d, *keys, default=None):
    """Defensive nested get for dicts."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d

def load_matches():
    if not os.path.exists(MATCHES_CSV):
        # If matches CSV not present, return empty dataframe
        return pd.DataFrame()
    return pd.read_csv(MATCHES_CSV)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_team_strengths(matches_df):
    # compute opponent strength as goals_conceded_per_match for the team across the tournament (basic)
    teams = {}
    if matches_df.empty:
        return {}
    for _, row in matches_df.iterrows():
        # flexible column names
        home = (row.get("home_team") or row.get("home_team_name") or row.get("home") or
                row.get("home_team_country") or row.get("home_team_id") or row.get("home_team_name"))
        away = (row.get("away_team") or row.get("away_team_name") or row.get("away") or
                row.get("away_team_country") or row.get("away_team_id") or row.get("away_team_name"))

        # try score columns in flexible ways
        home_goals = None
        away_goals = None
        for key in ("home_score", "home_team_score", "home_team_goals", "homeGoals"):
            if key in row and not pd.isna(row[key]):
                home_goals = int(row[key]); break
        for key in ("away_score", "away_team_score", "away_team_goals", "awayGoals"):
            if key in row and not pd.isna(row[key]):
                away_goals = int(row[key]); break
        # fallback to zeros if missing
        if home_goals is None:
            home_goals = 0
        if away_goals is None:
            away_goals = 0

        if pd.isna(home) or pd.isna(away):
            # skip malformed rows
            continue
        teams.setdefault(home, {"conceded": 0, "matches": 0})
        teams.setdefault(away, {"conceded": 0, "matches": 0})
        teams[home]["conceded"] += away_goals
        teams[home]["matches"] += 1
        teams[away]["conceded"] += home_goals
        teams[away]["matches"] += 1

    team_strength = {}
    for t, v in teams.items():
        if v["matches"] > 0:
            team_strength[t] = v["conceded"] / v["matches"]
        else:
            team_strength[t] = None
    return team_strength

def compute_minutes_played(starting_players, subs_events):
    """
    starting_players: dict player_id -> True means started
    subs_events: list of substitution events with keys:
        {'player_in_id','player_out_id','minute'}
    Returns dict player_id -> minutes (approx; assumes 90 if not subbed out/in recorded)
    """
    minutes = {}
    for pid in starting_players:
        minutes[pid] = 90

    for s in subs_events:
        out = s.get("player_out_id")
        inn = s.get("player_in_id")
        minute = s.get("minute", 90)
        try:
            minute = int(minute)
        except Exception:
            minute = 90
        if out is not None:
            minutes[str(out)] = max(0, minute)
        if inn is not None:
            # if player came in at minute m, minutes played ~ (90 - m)
            minutes[str(inn)] = max(0, 90 - minute)
    return minutes

def normalize_lineups(lineups_raw):
    """
    Accepts lineups that are either:
      - a list of two team dicts (older format), or
      - a dict with two team entries (your dataset)
    Returns a list of two 'side' dicts, each with keys like 'team' and 'lineup' or similar.
    """
    if isinstance(lineups_raw, list):
        return lineups_raw
    if isinstance(lineups_raw, dict):
        # try to extract the two team dicts (values)
        vals = list(lineups_raw.values())
        # sometimes values may include metadata; ensure each val is a dict with players
        sides = []
        for v in vals:
            if isinstance(v, dict):
                sides.append(v)
            elif isinstance(v, list) and v:
                # e.g., {'home': [players...], 'away': [players...]} unusual, wrap
                sides.append({"lineup": v})
        return sides
    # unknown shape -> return empty list
    return []

def extract_lineup_players(side):
    """
    Given one side structure, try to return list of player dicts with id, name, position.
    Defensive: returns [] if the side is not a dict or doesn't contain a players list.
    """
    # if side is not a dict, bail out quickly
    if not isinstance(side, dict):
        return []

    players = []
    # Common places for players:
    for candidate in ("lineup", "starting_lineup", "startXI", "players", "lineupPlayers", "startXIPlayers"):
        lst = safeget(side, candidate)
        if isinstance(lst, list) and lst:
            players = lst
            break
    # Some datasets store players directly as top-level list inside side (rare)
    if not players:
        # find any list-valued key that looks like players (heuristic)
        for k, v in side.items():
            if isinstance(v, list) and len(v) > 3:
                # Heuristic: lists longer than 3 are likely player lists
                players = v
                break
    # If still empty, return empty list rather than attempting .items() elsewhere
    return players

def find_player_id_from_playerobj(p):
    """
    Given a player entry from lineup or event, try many fields to find stable player id (as string).
    """
    if p is None:
        return None
    if isinstance(p, dict):
        for key in ("player",):
            if key in p and isinstance(p[key], dict):
                candidate = p[key].get("id") or p[key].get("player_id") or p[key].get("wyId") or p[key].get("playerId")
                if candidate:
                    return str(candidate)
        for key in ("id", "player_id", "playerId", "wyId"):
            if key in p:
                return str(p[key])
        # sometimes player name only
        for key in ("name", "player_name", "playerName"):
            if key in p:
                return str(p[key])
    elif isinstance(p, (int, float)):
        return str(int(p))
    elif isinstance(p, str):
        return p
    return None

def parse_events_for_match(match_id, events, lineups_raw):
    """
    Returns per-player stats dict keyed by player_id (string).
    Robust to different 'player' shapes in events and different lineup shapes.
    """
    per_player = defaultdict(lambda: defaultdict(int))
    # normalize lineup sides into list of two dicts
    sides = normalize_lineups(lineups_raw)
    starting_players = {}
    # build starting players map (strings)
    for side in sides:
        pls = extract_lineup_players(side)
        if isinstance(pls, list):
            for p in pls:
                pid = find_player_id_from_playerobj(p)
                if pid:
                    starting_players[pid] = True

    subs_events = []
    for ev in events:
        # robust event type and minute
        etype = safeget(ev, "type", "name") or ev.get("type_name") or ev.get("type")
        minute = ev.get("minute") or safeget(ev, "minute") or 90

        # substitution detection (gather in normalized form)
        if isinstance(etype, str) and "sub" in etype.lower():
            sub = safeget(ev, "substitution") or {}
            player_in = find_player_id_from_playerobj(safeget(sub, "player_in") or safeget(ev, "player_in") or safeget(ev, "playerIn") or safeget(ev, "player_in_id") or safeget(ev, "playerInId"))
            player_out = find_player_id_from_playerobj(safeget(sub, "player_out") or safeget(ev, "player_out") or safeget(ev, "playerOut") or safeget(ev, "player_out_id") or safeget(ev, "playerOutId"))
            if player_in or player_out:
                subs_events.append({"player_in_id": player_in, "player_out_id": player_out, "minute": minute})

        # find the actor player(s) for this event
        pids = []
        # 1) event 'player' field might be dict or string
        p_field = ev.get("player")
        if isinstance(p_field, dict):
            pid = find_player_id_from_playerobj(p_field)
            if pid:
                pids.append(pid)
        elif isinstance(p_field, (int, float, str)):
            pids.append(str(p_field))

        # 2) direct keys
        for key in ("player_id", "playerId", "player_id.1", "playerIdRef"):
            if key in ev and ev[key] is not None:
                pids.append(str(ev[key]))

        # 3) related_players
        rel = ev.get("related_players") or ev.get("relatedPlayers") or ev.get("players")
        if isinstance(rel, list):
            for r in rel:
                if isinstance(r, dict):
                    pid = find_player_id_from_playerobj(r)
                    if pid:
                        pids.append(pid)

        # 4) shot assist nested players
        if isinstance(ev.get("shot"), dict):
            assist = safeget(ev, "shot", "assist")
            if isinstance(assist, dict):
                pid = find_player_id_from_playerobj(assist.get("player") or assist)
                if pid:
                    pids.append(pid)

        # dedupe
        pids = list(dict.fromkeys([p for p in pids if p is not None]))

        if not pids:
            # skip events with no player candidate
            continue

        # choose the first pid as the actor for simple aggregation
        pid = pids[0]

        # Shots & goals
        if (isinstance(etype, str) and "shot" in etype.lower()) or ev.get("type_name") == "Shot":
            per_player[pid]["shots"] += 1
            outcome = safeget(ev, "shot", "outcome", "name") or ev.get("shot_outcome_name")
            if isinstance(outcome, str) and outcome.lower() == "goal":
                per_player[pid]["goals"] += 1
            # shots_on_target: try several possible fields
            if safeget(ev, "shot", "on_target") or safeget(ev, "shot", "shot_on_target") or ev.get("shot_on_target"):
                per_player[pid]["shots_on_target"] += 1

            # assist sometimes in shot.assist.player
            if safeget(ev, "shot", "assist"):
                per_player[pid]["assists"] += 1

        # Passes -> key passes
        if (isinstance(etype, str) and "pass" in etype.lower()) or ev.get("type_name") == "Pass":
            per_player[pid]["passes"] += 1
            if safeget(ev, "pass", "key_pass") or safeget(ev, "pass", "is_key_pass") or safeget(ev, "pass", "key_pass", "name") or ev.get("pass_goal_assist"):
                per_player[pid]["key_passes"] += 1

        # Tackles
        if (isinstance(etype, str) and "tackle" in etype.lower()) or ev.get("type_name") == "Tackle":
            per_player[pid]["tackles"] += 1

        # Cards
        if (isinstance(etype, str) and "card" in etype.lower()) or ev.get("type_name") == "Card":
            cardtype = safeget(ev, "card", "type") or safeget(ev, "card", "card_type") or ev.get("card_type") or ev.get("card")
            if cardtype and "yellow" in str(cardtype).lower():
                per_player[pid]["yellow_cards"] += 1
            if cardtype and "red" in str(cardtype).lower():
                per_player[pid]["red_cards"] += 1

    # minutes map from starting + subs
    minutes_map = compute_minutes_played(starting_players, subs_events)

    # ensure starters with zero events appear
    for pid in starting_players:
        if pid not in per_player:
            per_player[pid] = defaultdict(int)
            per_player[pid]["minutes"] = minutes_map.get(pid, 0)

    # attach minutes to players with events
    for pid in list(per_player.keys()):
        per_player[pid]["minutes"] = per_player[pid].get("minutes", minutes_map.get(pid, 0))

    return per_player

def compute_fantasy_points(row, position=None):
    # position is optional; default to 'M' if missing
    pos = (position or "M")[0].upper()
    pts = 0
    pts += int(row.get("goals", 0)) * SCORING["goal"].get(pos, SCORING["goal"]["M"])
    pts += int(row.get("assists", 0)) * SCORING["assist"]
    pts += int(row.get("shots_on_target", 0)) * SCORING["shot_on_target"]
    pts += int(row.get("key_passes", 0)) * SCORING["key_pass"]
    pts += int(row.get("tackles", 0)) * SCORING["tackle"]
    pts += int(row.get("yellow_cards", 0)) * SCORING["yellow_card"]
    pts += int(row.get("red_cards", 0)) * SCORING["red_card"]
    return pts

def extract_teamname_from_matchmeta(match_meta):
    # try multiple column names
    for k in ("home_team", "home_team_name", "home", "home_team_country", "home_team_id", "home_team_name"):
        if k in match_meta and not pd.isna(match_meta[k]):
            return match_meta[k]
    return None

def main():
    matches_df = load_matches()
    team_strength = compute_team_strengths(matches_df)
    out_rows = []
    event_files = sorted(glob.glob(os.path.join(RAW_DIR, "events_*.json")))

    for ef in tqdm(event_files, desc="processing matches"):
        basename = os.path.basename(ef)
        try:
            match_id = int(basename.split("_")[1].split(".")[0])
        except Exception:
            match_id = basename.replace("events_", "").replace(".json", "")

        events = load_json(ef)
        # events should be a list (per debug); if dict with 'events' key, use that
        if isinstance(events, dict) and "events" in events and isinstance(events["events"], list):
            events_list = events["events"]
        elif isinstance(events, list):
            events_list = events
        else:
            # fallback
            events_list = [events] if events else []

        # load lineup (may be dict or list)
        lineup_path = os.path.join(RAW_DIR, f"lineups_{match_id}.json")
        lineups = []
        if os.path.exists(lineup_path):
            try:
                lineups = load_json(lineup_path)
            except Exception:
                lineups = []

        # parse events for per-player stats
        perplayer = parse_events_for_match(match_id, events_list, lineups)

        # try best-effort match meta (if present in matches_df)
        match_meta = {}
        if not matches_df.empty:
            try:
                mrow = matches_df[matches_df.get("match_id", matches_df.columns[0]) == match_id]
                if mrow.shape[0] == 0:
                    # alternative attempt: some CSVs use 'id' or 'match_id' as string
                    if "id" in matches_df.columns:
                        mrow = matches_df[matches_df['id'] == match_id]
                if not mrow.empty:
                    match_meta = mrow.iloc[0].to_dict()
            except Exception:
                match_meta = {}

        # extract home/away and scores from match_meta if possible
        home_team = match_meta.get("home_team") or match_meta.get("home_team_name") or match_meta.get("home") or None
        away_team = match_meta.get("away_team") or match_meta.get("away_team_name") or match_meta.get("away") or None
        home_goals = match_meta.get("home_score") or match_meta.get("home_team_score") or None
        away_goals = match_meta.get("away_score") or match_meta.get("away_team_score") or None

        # if scores missing, compute from events (approx)
        if home_goals is None or away_goals is None:
            home_goals = 0
            away_goals = 0
            for ev in events_list:
                # shot goal detection robustly
                if isinstance(ev, dict):
                    evtype = safeget(ev, "type", "name") or ev.get("type_name") or ev.get("type")
                    if isinstance(evtype, str) and "shot" in evtype.lower():
                        outcome = safeget(ev, "shot", "outcome", "name") or ev.get("shot_outcome_name")
                        if isinstance(outcome, str) and outcome.lower() == "goal":
                            teamname = safeget(ev, "team", "name") or ev.get("team_name") or safeget(ev, "team")
                            if teamname == home_team:
                                home_goals += 1
                            else:
                                away_goals += 1

        # build pid->team mapping from lineups (defensive)
        pid_to_team = {}
        sides = []
        if isinstance(lineups, dict):
            # add dictionary values
            sides = list(lineups.values())
        elif isinstance(lineups, list):
            sides = lineups
        # each side may be a dict containing team info and player lists
        for side in sides:
            if not isinstance(side, dict):
                continue
            teamname = safeget(side, "team", "name") or side.get("team_name") or side.get("team")
            players = extract_lineup_players(side)
            if isinstance(players, list):
                for p in players:
                    pid = find_player_id_from_playerobj(p)
                    if pid:
                        pid_to_team[pid] = teamname

        # iterate perplayer results and create output rows
        for pid, stats in perplayer.items():
            # team/opponent/home determination
            teamname = pid_to_team.get(pid)
            if teamname is not None:
                if home_team and teamname == home_team:
                    opponent = away_team
                    goals_conceded = away_goals
                    home_flag = 1
                elif away_team and teamname == away_team:
                    opponent = home_team
                    goals_conceded = home_goals
                    home_flag = 0
                else:
                    opponent = None
                    goals_conceded = None
                    home_flag = None
            else:
                opponent = None
                goals_conceded = None
                home_flag = None

            # attempt to find position from lineup (search sides)
            position = None
            for side in sides:
                players = extract_lineup_players(side)
                if isinstance(players, list):
                    for p in players:
                        pid_cand = find_player_id_from_playerobj(p)
                        if pid_cand == pid:
                            # potential position fields
                            position = safeget(p, "position", "name") or p.get("position") or p.get("player_position") or p.get("role")
            # compute fantasy points
            fantasy_raw = compute_fantasy_points(stats, position=position)
            clean_sheet_bonus = 0
            pos0 = (position or "M")[0].upper() if position else "M"
            if pos0 in ("D", "G", "GK") and int(stats.get("minutes", 0)) >= 60 and goals_conceded == 0:
                clean_sheet_bonus = SCORING["clean_sheet_GK"] if pos0 in ("GK", "G") else SCORING["clean_sheet_DEF"]

            total_pts = fantasy_raw + clean_sheet_bonus

            out_rows.append({
                "match_id": match_id,
                "player_id": str(pid),
                "team": teamname,
                "opponent": opponent,
                "home": home_flag,
                "position": position,
                "minutes": int(stats.get("minutes", 0)) if stats.get("minutes", None) is not None else None,
                "goals": int(stats.get("goals", 0)),
                "assists": int(stats.get("assists", 0)),
                "shots_on_target": int(stats.get("shots_on_target", 0)),
                "key_passes": int(stats.get("key_passes", 0)),
                "tackles": int(stats.get("tackles", 0)),
                "yellow_cards": int(stats.get("yellow_cards", 0)),
                "red_cards": int(stats.get("red_cards", 0)),
                "fantasy_points": total_pts,
                "opponent_strength": team_strength.get(opponent, None)
            })

    df_out = pd.DataFrame(out_rows)
    if df_out.empty:
        print("No player rows found — df_out is empty.")
    else:
        # coerce player_id to string
        df_out["player_id"] = df_out["player_id"].astype(str)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}. Rows: {len(df_out)}")

if __name__ == "__main__":
    main()
