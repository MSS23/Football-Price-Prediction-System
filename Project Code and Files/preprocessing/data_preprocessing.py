import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from prefect import flow, task


def DropColumnsAndNoValuesInColumns(dataframe):
    dataframe.drop(
        [
            "real_face",
            "player_tags",
            "sofifa_id",
            "player_url",
            "long_name",
            "dob",
            "league_name",
            "international_reputation",
            "player_tags",
            "loaned_from",
            "team_jersey_number",
            "joined",
            "nation_position",
            "international_reputation",
            "nation_jersey_number",
            "nationality",
            "body_type",
            "player_traits",
            "lwb",
            "rcm",
            "cm",
            "rdm",
            "lb",
            "rwb",
            "rw",
            "lm",
            "lw",
            "lcm",
            "lcb",
            "cb",
            "rcb",
            "rm",
            "st",
            "cf",
            "lf",
            "rf",
            "lam",
            "cam",
            "ram",
            "ls",
            "rs",
            "rb",
            "ldm",
            "cdm",
            "team_position",
            "gk_diving",
            "gk_handling",
            "gk_kicking",
            "gk_reflexes",
            "gk_positioning",
        ],
        axis=1,
        inplace=True,
    )
    dataframe.dropna(subset=["player_positions", "value_eur"], inplace=True)

    dataframe["release_clause_eur"].fillna(0, inplace=True)
    dataframe["league_rank"].fillna(5, inplace=True)
    dataframe.drop(dataframe.index[dataframe["value_eur"] == 0], inplace=True)
    dataframe.dropna(subset=["club_name", "contract_valid_until"], inplace=True)
    dataframe.drop(columns=["defending_marking"], inplace=True)

    return dataframe


def AddingWorkRates(dataframe):
    # fill in the missing values for attacking and defensive work rate
    dataframe["attacking_work_rate"] = dataframe["work_rate"].map(
        lambda x: x.split("/")[0]
    )
    dataframe["defensive_work_rate"] = dataframe["work_rate"].map(
        lambda x: x.split("/")[1]
    )
    # drop the original work rate column

    # clean up the work rates too
    dataframe.drop(columns=["work_rate"], inplace=True)
    cleanup_work_rates = {
        "attacking_work_rate": {"Low": 1, "Medium": 2, "High": 3},
        "defensive_work_rate": {"Low": 1, "Medium": 2, "High": 3},
    }
    dataframe = dataframe.replace(cleanup_work_rates)

    path = "C:/Users/msidh/Documents/{file_name}.csv"
    return dataframe


def NormaliseData(dataframe):
    f"figure_min_temp_{dataframe}"
    # Columns that should not be normalized
    columns_not_being_normalised = [
        "short_name",
        "club_name",
        "player_positions",
        "wage_eur",
        "preferred_foot",
        "value_eur",
    ]

    # Select and preserve the columns that should not be normalized
    dataframe_not_normalized = dataframe[columns_not_being_normalised].copy()

    # Select the columns that should be normalized
    dataframe_to_normalize = dataframe.drop(columns=columns_not_being_normalised)

    # Apply normalization
    normalisation = MinMaxScaler()
    array_normalised = normalisation.fit_transform(dataframe_to_normalize)

    # Create a dataframe from the normalized array with the correct column names
    dataframe_normalized = pd.DataFrame(
        array_normalised, columns=dataframe_to_normalize.columns, index=dataframe.index
    )

    # Concatenate the non-normalized and normalized dataframes
    combined_df = pd.concat([dataframe_not_normalized, dataframe_normalized], axis=1)

    return combined_df


def get_player_names(dataframe):
    """
    Extracts the names of all players from the dataframe.

    Parameters:
    - dataframe: A Pandas DataFrame containing player data.

    Returns:
    - A list of player names.
    """
    return dataframe["short_name"].dropna().unique().tolist()


def cleanupPlayerPositions(dataframe):
    dataframe["player_positions_one"] = dataframe["player_positions"].map(
        lambda x: x.split(",")[0]
    )
    dataframe.drop(columns=["player_positions"], inplace=True)
    cleanup_player_positions = {
        "player_positions_one": {
            "GK": 1,
            "LWB": 2,
            "LB": 3,
            "CB": 4,
            "RWB": 5,
            "RB": 6,
            "CDM": 7,
            "CM": 8,
            "LM": 9,
            "RM": 10,
            "CAM": 11,
            "CF": 12,
            "LW": 13,
            "RW": 14,
            "ST": 15,
        }
    }

    dataframe = dataframe.replace(cleanup_player_positions)
    list_of_positions = dataframe.player_positions_one.unique().tolist()

    dataframe.drop(
        columns=["club_name", "league_rank", "release_clause_eur"], inplace=True
    )
    cleanup_preferred_foot = {"preferred_foot": {"Left": 0, "Right": 1}}

    dataframe = dataframe.replace(cleanup_preferred_foot)
    finalDataframe = dataframe

    return finalDataframe


def SplitDataframe(dataframe):
    # Split the dataframe into goalkeepers and outfield players
    goalkeepers_dataframe = dataframe[dataframe["player_positions_one"] == 1]

    # For goalkeepers, drop all outfield player stats
    goalkeeping_stats_to_remove = [
        "shooting",
        "pace",
        "passing",
        "dribbling",
        "defending",
        "physic",
        "skill_moves",
    ]
    goalkeepers_dataframe.drop(
        columns=goalkeeping_stats_to_remove, inplace=True, errors="ignore"
    )

    # For outfield players, drop any goalkeeping stat related columns
    outfield_related_stats_to_remove = [
        "gk_speed",
        "gk_diving",
        "gk_handling",
        "gk_kicking",
        "gk_positioning",
        "gk_reflexes",
    ]
    outfield_dataframe = dataframe[dataframe["player_positions_one"] != 1]
    outfield_dataframe.drop(
        columns=outfield_related_stats_to_remove, inplace=True, errors="ignore"
    )

    # Drop rows with NaN values if any
    outfield_dataframe.dropna(inplace=True)
    goalkeepers_dataframe.dropna(inplace=True)

    return goalkeepers_dataframe, outfield_dataframe


def dropPlayerNamesFromDataframe(playerName, outfield_dataframe, goalkeepers_dataframe):
    # Drop the 'short_name' column from both dataframes
    outfield_dataframe.drop(columns=["short_name"], inplace=True, errors="ignore")
    goalkeepers_dataframe.drop(columns=["short_name"], inplace=True, errors="ignore")
    return outfield_dataframe, goalkeepers_dataframe


def playerIndex(players, playerName):
    for i, name in enumerate(players):
        if name == playerName:
            return i
    return None  # Return None if the player is not found


def selectedDataframe(playerName, outfield_dataframe, goalkeepers_dataframe):
    print(outfield_dataframe.columns)
    if playerName in outfield_dataframe["short_name"].values:
        return outfield_dataframe
    elif playerName in goalkeepers_dataframe["short_name"].values:
        return goalkeepers_dataframe
    else:
        raise ValueError(f"Player name {playerName} not found in any dataframe")


def removePlayerAndPlayerNamesFromDataframe(playerName, dataframe):
    """
    Removes a specific player's row from the dataframe and returns it.
    """
    rowForPlayer = dataframe.loc[dataframe["short_name"] == playerName].copy()
    return rowForPlayer


def removePlayerFromDataframe(playerName, dataframe):
    """
    Removes a specific player's row from the dataframe and also drops non-numeric columns like 'short_name'.
    """
    playerNameIndex = dataframe.index[dataframe["short_name"] == playerName].tolist()
    dataframe_cleaned = dataframe.drop(playerNameIndex)

    # Drop the 'short_name' column to prevent issues with non-numeric data in ML models
    if "short_name" in dataframe_cleaned.columns:
        dataframe_cleaned = dataframe_cleaned.drop(columns=["short_name"])

    return dataframe_cleaned


def percentageDiff(actualValue, predictedValue):
    if actualValue == 0:
        raise ValueError("Actual value is zero, cannot compute percentage difference")
    return ((predictedValue - actualValue) / actualValue) * 100


def playerValue(playerName, dataframe):
    if playerName not in dataframe["short_name"].values:
        raise ValueError(f"Player name {playerName} not found in the dataframe")
    return dataframe.loc[dataframe["short_name"] == playerName, "value_eur"].iloc[0]
