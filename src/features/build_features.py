"""
Script to build features from the raw data
"""
from __future__ import annotations
import pickle
import pandas as pd
import numpy as np
from statistics import mean
from pathlib import Path
import os
import sys

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
            self.raw_dataframe = pd.read_csv(
                "data/interim/{0}/{0}.csv".format(subject_name)
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
                "data/interim/{0}".format(subject_name), "{}.csv".format(subject_name)
            )
            # save the csv file
            self.raw_dataframe.to_csv(path_or_buf=fullname)

    def split_data(
        self,
        training_split: float,
        start_num: float = 0,
    ):
        """
        Function to split data into training sets
        """
        # calculate end row number at which split will take place
        split_num = int(self.raw_dataframe.shape[0] * training_split)
        # do the actual split and drop the index
        train, test = (
            self.raw_dataframe.iloc[start_num:split_num, 1:],
            self.raw_dataframe.iloc[split_num:, 1:],
        )
        # return arrays of the values
        return np.array(train.values), np.array(test.values)

    def to_supervised(self, train: np.array, n_steps_in: int, n_steps_out: int):

        """
        Function to shape data into a shape suitable for
        supervised learning

        Modified form of function from Deep Learning for Time Series Forecasting by Jason Brownlee
        """
        X, y = [], []
        in_start = 0
        # step over entire data 1 step at a time
        for _ in range(len(train)):
            # define end of input sequence
            in_end = in_start + n_steps_in
            out_end = in_end - n_steps_out
            # ensure we have enough data
            if out_end < len(train):
                # extract features
                x_input = train[in_start:in_end, :-1]
                X.append(x_input)
                # extract heart rate info
                y.append(train[out_end, -1])
            # move along one time step
            in_start += 1
        return np.array(X), np.array(y)

    def moving_windows(self, window_shift: int):
        """
        Function to generate moving windows of data for training
        """
        pass


if __name__ == "__main__":

    data = DataBlock("S1", "data/raw/")
    train, test = data.split_data(training_split=0.5)
    X, y = data.to_supervised(train, 8, 1)
    print(data.raw_dataframe.head())
    print(X[0])
    print(y[0])
