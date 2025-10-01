# scripts/debug_events.py
"""
Self-contained diagnostic for StatsBomb events & lineups in data/raw.
Run: python scripts/debug_events.py
"""
import os
import glob
import json
from collections import defaultdict

RAW_DIR = "data/raw"
EVENT_GLOB = os.path.join(RAW_DIR, "events_*.json")
LINEUP_TEMPLATE = os.path.join(RAW_DIR, "lineups_{}.json")

def safeget(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def summarize_value(v, maxchars=300):
    try:
        s = json.dumps(v, default=str)
    except Exception:
        s = str(v)
    return s if len(s) <= maxchars else s[:maxchars] + "...[truncated]"

def find_players_in_event(ev):
    """
    Try many plausible spots for a player id/name in an event object.
    Return list of candidate ids/names (strings).
    """
    cands = []
    # 1) player field (could be dict, string, or id)
    p = ev.get("player")
    if isinstance(p, dict):
        pid = p.get("id") or p.get("player_id") or p.get("wyId") or p.get("playerId")
        pname = p.get("name") or p.get("player_name")
        if pid is not None:
            cands.append(str(pid))
        elif pname is not None:
            cands.append(str(pname))
    elif isinstance(p, (int, str)):
        cands.append(str(p))

    # 2) direct keys
    for k in ("player_id", "playerId", "playerIdRef", "player_name", "playerName"):
        if k in ev:
            cands.append(str(ev[k]))

    # 3) nested shot/assist structures
    if isinstance(ev.get("shot"), dict):
        assist = safeget(ev, "shot", "assist")
        if isinstance(assist, dict):
            if "player_id" in assist:
                cands.append(str(assist["player_id"]))
            if "id" in assist:
                cands.append(str(assist["id"]))
            if "player" in assist and isinstance(assist["player"], dict):
                aid = assist["player"].get("id") or assist["player"].get("player_id")
                if aid:
                    cands.append(str(aid))

    # 4) related_players
    rel = ev.get("related_players") or ev.get("relatedPlayers") or ev.get("players")
    if isinstance(rel, list):
        for r in rel:
            if isinstance(r, dict):
                if "player_id" in r:
                    cands.append(str(r["player_id"]))
                elif "id" in r:
                    cands.append(str(r["id"]))
                elif "player" in r and isinstance(r["player"], dict):
                    rid = r["player"].get("id") or r["player"].get("player_id")
                    if rid:
                        cands.append(str(rid))

    # 5) substitution fields
    for sk in ("player_in", "player_in_id", "playerIn", "playerInId"):
        val = safeget(ev, sk)
        if val is not None:
            cands.append(str(val))
    for sk in ("player_out", "player_out_id", "playerOut", "playerOutId"):
        val = safeget(ev, sk)
        if val is not None:
            cands.append(str(val))

    # dedupe
    return list(dict.fromkeys([c for c in cands if c is not None and c != ""]))

def simple_perplayer_agg(events):
    """
    Very permissive per-player aggregator: returns dict pid->counts of
    common event types seen (goal, shot, pass, card, tackle).
    """
    per = defaultdict(lambda: defaultdict(int))
    for ev in events:
        pids = find_players_in_event(ev)
        if not pids:
            continue
        # pick first candidate as actor for simple agg
        pid = pids[0]
        ev_type = None
        # try common type spots
        ev_type = safeget(ev, "type", "name") or ev.get("type_name") or ev.get("type")
        # Shots & goals
        if isinstance(ev_type, str) and "shot" in ev_type.lower():
            per[pid]["shots"] += 1
            outcome = safeget(ev, "shot", "outcome", "name") or ev.get("shot_outcome_name")
            if isinstance(outcome, str) and outcome.lower() == "goal":
                per[pid]["goals"] += 1
        # Passes
        if isinstance(ev_type, str) and "pass" in ev_type.lower():
            per[pid]["passes"] += 1
            if safeget(ev, "pass", "key_pass") or safeget(ev, "pass", "is_key_pass"):
                per[pid]["key_passes"] += 1
        # Tackles
        if isinstance(ev_type, str) and "tackle" in ev_type.lower():
            per[pid]["tackles"] += 1
        # Cards
        if isinstance(ev_type, str) and "card" in ev_type.lower():
            cardt = safeget(ev, "card", "type") or ev.get("card_type") or ev.get("card")
            if cardt and "yellow" in str(cardt).lower():
                per[pid]["yellow_cards"] += 1
            if cardt and "red" in str(cardt).lower():
                per[pid]["red_cards"] += 1
    return per

def main():
    files = sorted(glob.glob(EVENT_GLOB))
    print("Found event files:", len(files))
    if not files:
        print("No events files found under", EVENT_GLOB)
        return

    # inspect first 6 files (or fewer)
    for i, f in enumerate(files[:6], start=1):
        print("\n--- File", i, ":", os.path.basename(f), "---")
        try:
            data = load_json(f)
        except Exception as e:
            print("  FAILED to load JSON:", e)
            continue

        # show whether it's a list or dict and length
        if isinstance(data, list):
            print("  events structure: list, length:", len(data))
        elif isinstance(data, dict):
            print("  events structure: dict, keys:", list(data.keys())[:10])
            # maybe actual events are under a key like 'events'
            if "events" in data and isinstance(data["events"], list):
                print("   -> contains 'events' list length:", len(data["events"]))
                events = data["events"]
            else:
                # not list, try convert to list of single dict
                events = [data]
        else:
            print("  unknown events type:", type(data))
            events = []

        if isinstance(data, list):
            events = data

        print("  events count (assigned):", len(events))
        if len(events) == 0:
            continue

        # first event keys and truncated sample
        first_ev = events[0]
        print("  First event keys:", list(first_ev.keys())[:50])
        print("  First event sample (truncated):")
        print(summarize_value({k: first_ev[k] for k in list(first_ev.keys())[:20]}, maxchars=1000))

        # try find player ids in first 20 events
        pset = set()
        for ev in events[:20]:
            found = find_players_in_event(ev)
            if found:
                for fnd in found:
                    pset.add(fnd)
        print("  player-candidates found in first 20 events (sample up to 20):", list(pset)[:20], "total:", len(pset))

        # try simple per-player aggregation for this file
        per = simple_perplayer_agg(events)
        print("  simple per-player aggregator returned n players:", len(per))
        # print up to 8 players and their small stats
        cnt = 0
        for pid, stats in per.items():
            print("    pid:", pid, "stats:", dict(stats))
            cnt += 1
            if cnt >= 8:
                break

        # does a lineup file exist?
        try:
            mid = os.path.basename(f).split("_")[1].split(".")[0]
            lineup_path = LINEUP_TEMPLATE.format(mid)
            if os.path.exists(lineup_path):
                print("  lineup exists for match:", lineup_path)
                try:
                    lineup = load_json(lineup_path)
                    print("   lineup top-level type:", type(lineup), "len:" , len(lineup) if hasattr(lineup, "__len__") else "n/a")
                    # sample first lineup structure
                    print("   lineup sample keys (if list):", list(lineup[0].keys())[:20] if isinstance(lineup, list) and lineup else "n/a")
                except Exception as e:
                    print("   failed to load lineup:", e)
            else:
                print("  lineup missing for match id:", mid)
        except Exception as e:
            print("  couldn't parse match id from filename:", e)

    # extra: run simple_perplayer_agg on all files and report totals
    total_players = set()
    total_event_counts = 0
    for f in files:
        try:
            data = load_json(f)
            events = data if isinstance(data, list) else (data.get("events") if isinstance(data, dict) and isinstance(data.get("events"), list) else [data])
            per = simple_perplayer_agg(events)
            for pid in per.keys():
                total_players.add(pid)
            total_event_counts += len(events)
        except Exception:
            continue
    print("\nAcross all files: total events (approx):", total_event_counts, "unique player ids found (approx):", len(total_players))
    print("Sample player ids (up to 20):", list(total_players)[:20])

if __name__ == "__main__":
    main()
