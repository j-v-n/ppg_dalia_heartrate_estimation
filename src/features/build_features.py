"""
Script to build features from the raw data
"""
from __future__ import annotations
import pickle
from xmlrpc.client import Boolean
import pandas as pd
import numpy as np
from statistics import mean
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import os
import sys
import pprint

path_to_join = os.path.join(os.getcwd(), "src")
sys.path.insert(0, path_to_join)

from visualization.visualize import create_dataframe


ACTIVITY_DICT = {
    0: "transient",
    1: "sitting",
    2: "stairs",
    3: "table_soccer",
    4: "cycling",
    5: "driving",
    6: "lunch_break",
    7: "walking",
    8: "working",
}
BVP_FREQUENCY = 64
ACC_FREQUENCY = 32


class DataBlock:
    """
    Class to read data from .pkl file,
    create supervised learning features
    """

    def __init__(self, subject_name: str, filepath: Path):

        # check if a csv file exists
        if os.path.exists("data/interim/{0}/{0}.csv".format(subject_name)):
            # load the data into a dataframe
            self.raw_dataframe = pd.read_csv(
                "data/interim/{0}/{0}.csv".format(subject_name), index_col=0
            )
        else:
            # if not, create the dataframe and save to a csv file for later use
            self.raw_dataframe = create_dataframe(
                filepath=os.path.join(filepath, subject_name),
                subject_name=subject_name,
            )
            # replace gender strings with 0 or 1 -> 0 = male, 1 = female
            self.raw_dataframe["gender"] = np.where(
                self.raw_dataframe["gender"] == " m", 0, 1
            )
            # drop subject number from features
            self.raw_dataframe.drop("subject", axis=1, inplace=True)
            # check if folder exists for the subject
            if not os.path.exists("data/interim/{}".format(subject_name)):
                # if not, create one
                os.mkdir("data/interim/{0}".format(subject_name))
            # obtain full path for target csv file
            fullname = os.path.join(
                "data/interim/{0}".format(subject_name),
                "{}.csv".format(subject_name),
            )
            self.raw_dataframe["bmi"] = (
                self.raw_dataframe["weight"]
                * 10000
                / (self.raw_dataframe["height"] ** 2)
            )

            self.raw_dataframe = self.raw_dataframe[
                [
                    "bvp",
                    "acc_x",
                    "acc_y",
                    "acc_z",
                    "gender",
                    "age",
                    "sport",
                    "bmi",
                    "heart_rate",
                ]
            ]
            # save the dataframe to csv
            self.raw_dataframe.to_csv(path_or_buf=fullname)

        self.n_samples = self.raw_dataframe.shape[0]


def to_supervised(train: np.array, n_steps_in: int, n_steps_out: int):

    """
    Function to shape data into a shape suitable for
    supervised learning

    Modified form of function from Deep Learning for Time Series Forecasting by Jason Brownlee

    Args:
    train:np.array -> the training numpy array obtained from split_data method
    n_steps_in:int -> number of timesteps input considered
    n_steps_out:int -> number of timesteps of output considered

    Returns:
    X:np.array -> supervised learning style array of features
    y:np.array -> vector of heart-rate values
    """
    X, y = [], []
    in_start = 0
    # step over entire data 1 step at a time
    for i in range(len(train)):
        # define end of input sequence
        end_ix = n_steps_in + (i)
        # out_end = in_end - n_steps_out
        # ensure we have enough data
        if end_ix < len(train):
            # extract features
            x_input = train[i:end_ix, :-1]
            X.append(x_input)
            # extract heart rate info
            y.append(np.mean(train[i:end_ix, -1]))
        # move along one time step
        # in_start += 2

    return np.array(X), np.array(y)


def to_supervised_shuffled(train: np.array, n_steps_in: int, n_steps_out: int):

    """
    Function to shape data into a shape suitable for
    supervised learning

    Modified form of function from Deep Learning for Time Series Forecasting by Jason Brownlee

    Args:
    train:np.array -> the training numpy array obtained from split_data method
    n_steps_in:int -> number of timesteps input considered
    n_steps_out:int -> number of timesteps of output considered

    Returns:
    X:np.array -> supervised learning style array of features
    y:np.array -> vector of heart-rate values
    """
    X, y = [], []
    in_start = 0
    # step over entire data 1 step at a time
    for i in range(len(train)):
        # define end of input sequence
        end_ix = n_steps_in + (i)
        # out_end = in_end - n_steps_out
        # ensure we have enough data
        if end_ix < len(train):
            # extract features
            x_input = train[i:end_ix, :]
            X.append(x_input)
            # extract heart rate info
            # y.append(np.mean(train[i:end_ix, -1]))
        # move along one time step
        # in_start += 2
    X = np.array(X)
    np.random.shuffle(X)
    X_final = []
    for row in X:
        X_final.append(row[:, :-1])

    # y = []
    for row in X:
        y.append(np.mean(row[:, -1]))

    return np.array(X_final), np.array(y)


# def to_supervised_mlp(train: np.array, n_steps_in: int, n_steps_out: int):

#     """
#     Function to shape data into a shape suitable for
#     supervised learning

#     Modified form of function from Deep Learning for Time Series Forecasting by Jason Brownlee

#     Args:
#     train:np.array -> the training numpy array obtained from split_data method
#     n_steps_in:int -> number of timesteps input considered
#     n_steps_out:int -> number of timesteps of output considered

#     Returns:
#     X:np.array -> supervised learning style array of features
#     y:np.array -> vector of heart-rate values
#     """
#     X, y = [], []
#     in_start = 0
#     # step over entire data 1 step at a time
#     for _ in range(len(train)):
#         # define end of input sequence
#         in_end = in_start + n_steps_in
#         out_end = in_end - n_steps_out
#         # ensure we have enough data
#         if out_end < len(train):
#             # extract features
#             x_input = train[in_start:in_end, :-1]
#             X.append(x_input)
#             # extract heart rate info
#             y.append(train[out_end, -1])
#         # move along one time step
#         in_start += 1
#     return np.array(X), np.array(y)

if __name__ == "__main__":
    for i in range(1, 16):
        datablock = DataBlock(subject_name="S{}".format(i), filepath="data/raw/")
