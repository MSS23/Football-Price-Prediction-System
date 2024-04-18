from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
import numpy as np
import streamlit as st
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from preprocessing.data_preprocessing import (
    DropColumnsAndNoValuesInColumns,
    AddingWorkRates,
    NormaliseData,
    cleanupPlayerPositions,
    SplitDataframe,
    playerIndex,
    selectedDataframe,
    removePlayerAndPlayerNamesFromDataframe,
    removePlayerFromDataframe,
    percentageDiff,
    playerValue,
)


st.cache


def linearRegression(playerName, dataframe):
    np.random.seed(101)
    rowOfPlayer = removePlayerAndPlayerNamesFromDataframe(playerName, dataframe)
    dataframeCleaned = removePlayerFromDataframe(playerName, dataframe.copy())
    train, test = train_test_split(dataframeCleaned, test_size=0.3, random_state=42)
    x_train = train.drop("value_eur", axis=1)
    y_train = train["value_eur"]
    x_test = test.drop("value_eur", axis=1)
    y_test = test["value_eur"]
    regr = LinearRegression()
    regr.fit(x_train, y_train)
    regressionModel = regr.predict(
        rowOfPlayer.drop(["short_name", "value_eur"], axis=1)
    )
    return regressionModel


@st.cache
def knnBestValueGraph(x_train, y_train, x_test, y_test):
    rmseValues = []
    for k in range(1, 30):
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        error = sqrt(mean_squared_error(y_test, pred))
        rmseValues.append(error)
        # print("RMSE value for k= ", k, "is:", error)


@st.cache
def knnRegression(playerName, dataframe):
    np.random.seed(101)
    rowOfPlayer = removePlayerAndPlayerNamesFromDataframe(playerName, dataframe)
    dataframeCleaned = removePlayerFromDataframe(playerName, dataframe.copy())

    train, test = train_test_split(dataframeCleaned, test_size=0.3, random_state=42)
    x_train = train.drop("value_eur", axis=1)
    y_train = train["value_eur"]
    x_test = test.drop("value_eur", axis=1)
    y_test = test["value_eur"]

    x_dimension = np.arange(1, 100)
    params = {"n_neighbors": x_dimension, "weights": ["uniform", "distance"]}
    knn = KNeighborsRegressor(n_jobs=2)
    model = GridSearchCV(knn, params, cv=5)
    model.fit(x_train, y_train)

    predictedValue = model.predict(
        rowOfPlayer.drop(["short_name", "value_eur"], axis=1)
    )

    knnBestValueGraph(x_train, y_train, x_test, y_test)
    return model, model.best_params_, predictedValue


@st.cache
def decisionTreeRegression(playerName, dataframe, depthValue):
    np.random.seed(101)
    rowOfPlayer = removePlayerAndPlayerNamesFromDataframe(playerName, dataframe)
    dataframeCleaned = removePlayerFromDataframe(playerName, dataframe.copy())

    train, test = train_test_split(dataframeCleaned, test_size=0.3, random_state=42)
    x_train = train.drop("value_eur", axis=1)
    y_train = train["value_eur"]
    x_test = test.drop("value_eur", axis=1)
    y_test = test["value_eur"]

    tree_reg = DecisionTreeRegressor(max_depth=depthValue, random_state=42)
    tree_reg.fit(x_train, y_train)
    valueOfPlayer = tree_reg.predict(
        rowOfPlayer.drop(["short_name", "value_eur"], axis=1)
    )

    return valueOfPlayer


@st.cache
def randomForestRegression(playerName, dataframe, treeValue, maxFeatureVal):
    np.random.seed(101)
    rowOfPlayer = removePlayerAndPlayerNamesFromDataframe(playerName, dataframe)
    dataframeCleaned = removePlayerFromDataframe(playerName, dataframe.copy())

    train, test = train_test_split(dataframeCleaned, test_size=0.2, random_state=42)
    x_train = train.drop("value_eur", axis=1)
    y_train = train["value_eur"]
    x_test = test.drop("value_eur", axis=1)
    y_test = test["value_eur"]

    regressor = RandomForestRegressor(
        n_estimators=treeValue, max_features=maxFeatureVal, random_state=42
    )
    regressor.fit(x_train, y_train)
    predictedValue = regressor.predict(
        rowOfPlayer.drop(["short_name", "value_eur"], axis=1)
    )

    return predictedValue


@st.cache
def linearSVR(playerName, dataframe):
    np.random.seed(101)
    rowOfPlayer = removePlayerAndPlayerNamesFromDataframe(playerName, dataframe)
    dataframeCleaned = removePlayerFromDataframe(playerName, dataframe.copy())

    train, test = train_test_split(dataframeCleaned, test_size=0.3, random_state=42)
    x_train = train.drop("value_eur", axis=1)
    y_train = train["value_eur"]
    x_test = test.drop("value_eur", axis=1)
    y_test = test["value_eur"]

    regr = LinearSVR()
    regr.fit(x_train, y_train)
    predictedValue = regr.predict(rowOfPlayer.drop(["short_name", "value_eur"], axis=1))

    return predictedValue
