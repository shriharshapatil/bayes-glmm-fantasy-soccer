from statsbombpy import sb
import pandas as pd
# Example: getting all matches from the Women's World Cup 2023
matches = sb.matches(competition_id=43, season_id=106)

# Create an empty list to store event data
all_events = []

# Loop through each match to get event data
for match_id in matches['match_id']:
    events = sb.events(match_id=match_id)
    all_events.append(events)

# Concatenate all event data into a single DataFrame
event_df = pd.concat(all_events, ignore_index=True)