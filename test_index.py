# %% markdown
# # Test Notebook
# Run tests to make sur we are retrieving ans saving the stats per team
# and season

# %%
# Auto reload external librairies.
%load_ext autoreload
%autoreload 2

# %%
# Load necessary librairies ans classes
from SqliteHandler import SqliteHandler
from MongoHandler import MongoHandler
from WeatherHandler import WeatherHandler
from GameDataHandler import GameDataHandler
import config

# Retrieve / save data for season 2011
sql_handler = SqliteHandler(config.STAT_SQLITE_DB)
weather_handler = WeatherHandler(config.STAT_WEATHER_API_KEY)
mongo_handler = MongoHandler(config.STAT_MONGO_CONNECTION, config.STAT_MONGO_DB_NAME)
games_handler = GameDataHandler(sql_handler, weather_handler, mongo_handler)

# Load stats for 2011 season
games = games_handler.get_games_per_season(2011)

# %%
# check that we got data
print(games[0].away_team)
print(games[0].home_team)
print(games[0].ht_goals)
print(games[0].at_goals)
print(games[0].is_raining)

# Check Game class method has_team_won()
print(games[0].has_team_won('dsds'))
print(games[0].has_team_won('Bayern Munich'))

# %%
# Instantiate and retrieve/save stats for Bayern Munich team for 2011 season
from TeamSeasonStatistics import TeamSeasonStatistics

stats_bayern = TeamSeasonStatistics('Bayern Munich', 2011, games_handler, mongo_handler)

print(stats_bayern.get_statistics())

# Save into MongoDB
stats_bayern.save()
