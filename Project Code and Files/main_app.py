import os
import sys
import pandas as pd
import streamlit as st
from preprocessing.data_preprocessing import (
    DropColumnsAndNoValuesInColumns,
    AddingWorkRates,
    NormaliseData,
    cleanupPlayerPositions,
    SplitDataframe,
    get_player_names,
    selectedDataframe,
)
from models.models import (
    linearRegression,
    knnRegression,
    randomForestRegression,
    decisionTreeRegression,
    linearSVR,
)

# # Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
st.title("Football Player Price Prediction System - Dataset 2021")

filepath = os.path.join(os.path.dirname(__file__), "data", "footballData.csv")
playerNamesDataframe = pd.read_csv(filepath)

# # Load the data
# playerNamesDataframe = pd.read_csv(r"C:/Users/msidh/Documents/footballData.csv")

# Get player names using a function from 'data_preprocessing.py'
players = get_player_names(playerNamesDataframe)

# Preprocess the data
dataframe = DropColumnsAndNoValuesInColumns(playerNamesDataframe)
dataframe = AddingWorkRates(dataframe)
dataframe = NormaliseData(dataframe)
dataframe = cleanupPlayerPositions(dataframe)
goalkeepers_dataframe, outfield_dataframe = SplitDataframe(dataframe)


# Player selection UI
st.write("Player you want to find the price for")
playerName = st.selectbox("Select target player for price prediction", players)
dataframeForModel = selectedDataframe(
    playerName, goalkeepers_dataframe, outfield_dataframe
)

# Linear Regression
st.write("Linear Regression Model")
lrResult = linearRegression(playerName, dataframeForModel)
lrPrice = float(lrResult[0])
st.write(f"The predicted price of {playerName} is €{lrPrice:,.2f}")

# KNN Regression
st.write("KNN Regression Model")
knnResult = knnRegression(playerName, dataframeForModel)
knnPrice = float(knnResult[2][0])
st.write(f"The predicted price of {playerName} is €{knnPrice:,.2f}")
if st.checkbox("Best K Value"):
    st.write(f"Best K Value: {knnResult[1]}")

# Random Forest Regression
st.write("Random Forest Regression Model")

treeValue = st.slider(
    "Random Forest: How many trees do you want the forest to have", 0, 200, 100
)  # Default value set to 100
maxFeatureVal = st.selectbox(
    "Random Forest: What max feature value would you like to use?",
    ("log2", "sqrt"),
)

randomForestResult = randomForestRegression(
    playerName, dataframeForModel, treeValue, maxFeatureVal
)
rfPrice = float(randomForestResult[0])
st.write(f"The predicted price of {playerName} is €{rfPrice:,.2f}")

# Decision Tree Regression
st.write("Decision Tree Regression Model")
depthValue = st.slider(
    "Decision Tree: What tree depth do you want to use", 0, 50, 10
)  # Default value set to 10

decisionTreeReg = decisionTreeRegression(playerName, dataframeForModel, depthValue)
dtPrice = float(decisionTreeReg[0])
st.write(f"The predicted price of {playerName} is €{dtPrice:,.2f}")

# Linear SVR
st.write("Linear SVR Model")
linearSVRPrice = linearSVR(playerName, dataframeForModel)
svrPrice = float(linearSVRPrice[0])
st.write(f"The predicted price of {playerName} is €{svrPrice:,.2f}")
