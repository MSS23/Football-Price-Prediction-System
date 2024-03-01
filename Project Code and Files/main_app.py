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
    selectedDataframe
)
from models.models import (
    linearRegression,
    knnRegression,
    randomForestRegression,
    decisionTreeRegression,
    linearSVR
)

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the data
playerNamesDataframe = pd.read_csv(r'C:/Users/msidh/Documents/footballData.csv')

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
dataframeForModel = selectedDataframe(playerName, goalkeepers_dataframe, outfield_dataframe)

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

treeValue = st.slider('Random Forest: How many trees do you want the forest to have', 0, 200, 100)  # Default value set to 100
maxFeatureVal = st.selectbox('Random Forest: What max feature value would you like to use?', ('log2', 'auto', 'sqrt'))

randomForestResult = randomForestRegression(playerName, dataframeForModel, treeValue, maxFeatureVal)
rfPrice = float(randomForestResult[0])
st.write(f"The predicted price of {playerName} is €{rfPrice:,.2f}")

# Decision Tree Regression
st.write("Decision Tree Regression Model")
depthValue = st.slider('Decision Tree: What tree depth do you want to use', 0, 50, 10)  # Default value set to 10

decisionTreeReg = decisionTreeRegression(playerName, dataframeForModel, depthValue)
dtPrice = float(decisionTreeReg[0])
st.write(f"The predicted price of {playerName} is €{dtPrice:,.2f}")

# Linear SVR
st.write("Linear SVR Model")
linearSVRPrice = linearSVR(playerName, dataframeForModel)
svrPrice = float(linearSVRPrice[0])
st.write(f"The predicted price of {playerName} is €{svrPrice:,.2f}")

# import sys
# import os

# # Add the project root directory to the Python path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import pandas as pd
# import numpy as np  
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# from prefect import flow,task
# from preprocessing.data_preprocessing import (
#     DropColumnsAndNoValuesInColumns,
#     AddingWorkRates,
#     NormaliseData,
#     cleanupPlayerPositions,
#     SplitDataframe,
#     playerIndex,
#     removePlayerAndPlayerNamesFromDataframe,
#     removePlayerFromDataframe,
#     percentageDiff,
#     playerValue,
#     get_player_names,
#     selectedDataframe
# )

# from models.models import (linearRegression, knnRegression, randomForestRegression, decisionTreeRegression, linearSVR)  


# # Read the source of the data
# playerNamesDataframe = pd.read_csv(r'C:/Users/msidh/Documents/footballData.csv')

# # Use the function from 'data_preprocessing.py' to get player names
# players = get_player_names(playerNamesDataframe)

# # Preprocess the data
# dataframe = DropColumnsAndNoValuesInColumns(dataframe)
# dataframe = AddingWorkRates(dataframe)
# dataframe = NormaliseData(dataframe)
# dataframe = cleanupPlayerPositions(dataframe)
# goalkeepers_dataframe, outfield_dataframe = SplitDataframe(dataframe)

# # Now, your data is ready for the ML models

# #read the source of the data
# # dataframe = pd.read_csv(r'C:/Users/msidh/Documents/footballData.csv')
# playerNamesDataframe = pd.read_csv(r'C:/Users/msidh/Documents/footballData.csv')

# # Use the function from 'data_preprocessing.py'
# players = get_player_names(playerNamesDataframe)
#     # Widget calls moved outside the cached functions
# depthValue = st.slider('Decision Tree: What tree depth do you want to use', 0, 50, 1)
# treeValue = st.slider('Random Forest: How many trees do you want the forest to have', 0, 200, 1)
# maxFeatureVal = st.selectbox('Random Forest: What max feature value would you like to use?', ('log2', 'auto', 'sqrt'))


# st.write("Player you want to find the price for")
# playerName = st.selectbox("Select target player for price prediction", players)
# dataframeForModel = selectedDataframe(playerName, goalkeepers_dataframe, outfield_dataframe)

# st.write("Linear Regression Model")
# lrResult = linearRegression(playerName, dataframeForModel)
# lrPrice = float(lrResult[0]) if isinstance(lrResult, np.ndarray) else float(lrResult)
# st.write(f"The predicted price of {playerName} is €{lrPrice:,.2f} ")

# st.write("KNN Regression Model")
# knnResult = knnRegression(playerName, dataframeForModel)
# knnPrice = float(knnResult[2][0]) if isinstance(knnResult[2], np.ndarray) else float(knnResult[2])
# st.write(f"The predicted price of {playerName} is €{knnPrice:,.2f} ")
# kValue = st.checkbox("Best K Value")

# if kValue:
#     st.write(f"Best K Value: {knnResult[1]}")

# st.write("Random Forest Regression Model")
# randomForestResult = randomForestRegression(playerName, dataframeForModel)
# rfPrice = float(randomForestResult[0]) if isinstance(randomForestResult, np.ndarray) else float(randomForestResult)
# st.write(f"The predicted price of {playerName} is €{rfPrice:,.2f} ")

# st.write("Decision Tree Regression Model")
# decisionTreeReg = decisionTreeRegression(playerName, dataframeForModel)
# dtPrice = float(decisionTreeReg[0]) if isinstance(decisionTreeReg, np.ndarray) else float(decisionTreeReg)
# st.write(f"The predicted price of {playerName} is €{dtPrice:,.2f} ")

# st.write("Linear SVR Model")
# linearSVRPrice = linearSVR(playerName, dataframeForModel)
# svrPrice = float(linearSVRPrice[0]) if isinstance(linearSVRPrice, np.ndarray) else float(linearSVRPrice)
# st.write(f"The predicted price of {playerName} is €{svrPrice:,.2f}")


