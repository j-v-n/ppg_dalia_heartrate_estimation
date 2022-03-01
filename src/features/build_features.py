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

    def __init__(
        self, subject_name: str, filepath: Path, scale_and_encode: Boolean = True
    ):
        # check if we are scaling numerical columns and encoding categorical columns
        self.scaled_and_encoded = scale_and_encode
        # check if a csv file exists
        if os.path.exists("data/interim/{0}/{0}.csv".format(subject_name)):
            # load the data into a dataframe
            self.raw_dataframe = pd.read_csv(
                "data/interim/{0}/{0}.csv".format(subject_name)
            )
            # create a scaled and encoded numpy array
            self.scale_encode_func()
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
            # save the dataframe to csv
            self.raw_dataframe.to_csv(path_or_buf=fullname)
            # if we want scaling and encoding
            if self.scaled_and_encoded:
                # run the scaling and encoding method
                self.scale_encode_func()

    def split_data(
        self,
        training_split: float,
        start_num: int = 0,
    ):
        """
        Function to split data into training sets. If the scaled_and_encoded property is True,
        it will split the numpy array, otherwise splits the raw dataframe

        Args:
        training_split:float -> the fraction of data that will be used as training set
        start_num:int -> starting number for splitting

        Returns:
        train:np.array -> training numpy array
        test:np.array -> testing numpy array
        """
        # if we have scaled and encoded the data
        if self.scaled_and_encoded:
            split_num = int(self.scaled_array.shape[0] * training_split)
            # split into training and test sets
            train, test = (
                self.scaled_array[start_num:split_num, 1:],
                self.scaled_array[split_num:, 1:],
            )
            return train, test

        else:
            df = self.raw_dataframe
            # calculate end row number at which split will take place
            split_num = int(df.shape[0] * training_split)
            # do the actual split and drop the index
            train, test = (
                df.iloc[start_num:split_num, 1:],
                df.iloc[split_num:, 1:],
            )
            # return arrays of the values
            return np.array(train.values), np.array(test.values)

    def to_supervised(self, train: np.array, n_steps_in: int, n_steps_out: int):

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

    def scale_encode_func(self):
        """
        Function to scale numerical features and one hot encode categorical ones

        Returns:
        self.scaled_array:np.array -> a numpy array of scaled and encoded features
        """
        # the numeric features which are not dependent on the subject description
        numeric_features = [
            "bvp",
            "acc_x",
            "acc_y",
            "acc_z",
            "temp",
            "eda",
        ]
        # create a pipeline to do the transformation
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        # the categorical features not dependent on the subject description
        categorical_features = ["activity"]
        # create an encoder object to do the categorical transformation
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        # create a columntransformer object to do numerical and categorical columns separately.
        # we pass through the remainder for now
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="passthrough",
        )
        # fit the columntransformer to the dataframe, transform it and create an array
        self.scaled_array = preprocessor.fit_transform(self.raw_dataframe)


if __name__ == "__main__":

    data = DataBlock("S1", "data/raw/")
    train, test = data.split_data(training_split=0.7)
    X, y = data.to_supervised(train, 8, 1)
    pprint.pprint(data.raw_dataframe.head(9))
    pprint.pprint(X[1])
    pprint.pprint(y[1])
