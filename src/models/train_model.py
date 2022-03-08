from __future__ import annotations
import numpy as np
import pandas as pd
import os
import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .define_model import convLSTMModel
from src.features.build_features import DataBlock, to_supervised

from pickle import dump, load


def scaler(dataframe):
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
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="passthrough",
    )
    # fit the columntransformer to the dataframe
    preprocessor.fit(dataframe)
    # save the preprocessor as we will fit this scaler to validation and testing sets
    dump(preprocessor, open("models/scaler.pkl", "wb"))
    # return the transformed array
    return preprocessor.transform(dataframe)


class TrainModel:
    """
    Class to handle training using a convLSTM model
    """

    def __init__(
        self,
        train_subjects: list,
        test_subjects: list,
        valid_subjects: list,
        n_rows: int,
        n_cols: int,
        n_timesteps: int,
        n_features: int,
        n_conv_filters: int,
        n_dense_nodes: int,
        n_output_nodes: int,
        n_seq: int,
        batch_size: int,
        epochs: int,
    ):
        # define the model
        self.model = convLSTMModel(
            n_conv_filters=n_conv_filters,
            n_rows=n_rows,
            n_cols=n_cols,
            n_dense_nodes=n_dense_nodes,
            n_output_nodes=n_output_nodes,
            input_shape=(n_seq, 1, int(n_timesteps / n_seq), n_features),
        )
        # compile the model
        self.model.compile(loss="mse", metrics="mae", optimizer="adam")
        # define the train, test and valid subjects
        self.train_subjects = train_subjects
        self.test_subjects = test_subjects
        self.valid_subjects = valid_subjects
        # define the number of timesteps used in prediction
        self.timesteps = n_timesteps
        # define number of features used in the model
        self.features = n_features
        # define the length of each subsequence
        self.seq = n_seq
        # define the batch size
        self.batch_size = batch_size
        # define epochs
        self.epochs = epochs

    def load_data(self, subject_set: str = "train"):
        """
        Function to load data for training

        Args:
        """
        # create a list to hold dataframes
        list_of_dfs = []
        # load the data according to whether we are dealing with train,test or valid
        if subject_set == "train":
            for subject in self.train_subjects:
                data = DataBlock("S{}".format(subject), "data/raw/")
                df = data.raw_dataframe
                list_of_dfs.append(df)
        elif subject_set == "valid":
            for subject in self.valid_subjects:
                data = DataBlock("S{}".format(subject), "data/raw/")
                df = data.raw_dataframe
                list_of_dfs.append(df)
        else:
            for subject in self.test_subjects:
                data = DataBlock("S{}".format(subject), "data/raw/")
                df = data.raw_dataframe
                list_of_dfs.append(df)

        # concatenate the dataframes
        frames = pd.concat(list_of_dfs)
        # name the columns
        frames.columns = [
            "bvp",
            "acc_x",
            "acc_y",
            "acc_z",
            "temp",
            "eda",
            "heart_rate",
        ]
        # check if the scaler exists
        if os.path.exists("models/scaler.pkl"):
            # if so, load and use it
            saved_scaler = load(open("models/scaler.pkl", "rb"))
            scaled = saved_scaler.transform(frames)
        else:
            # else create and fit a new scaler and transform the dataframes
            scaled = scaler(frames)
        # convert into a supervised learning problem
        X, y = to_supervised(scaled, self.timesteps, 1)
        # reshape X into the format needed for convLSTM
        X = X.reshape(
            (
                X.shape[0],
                self.seq,
                1,
                int(self.timesteps / self.seq),
                self.features,
            )
        )
        return X, y

    def train(self):
        """
        Function to run training
        """
        # load training and validation data
        train_X, train_y = self.load_data(subject_set="train")
        valid_X, valid_y = self.load_data(subject_set="valid")
        # define callbacks
        # early stopping
        es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=10)
        # model checkpoint
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            "models/",
            monitor="val_mae",
            verbose=0,
            save_best_only=True,
            mode="max",
        )
        log_dir = "models/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard callback
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        # fit the model and save history
        self.history = self.model.fit(
            train_X,
            train_y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[es_callback, cp_callback, tb_callback],
            validation_data=(valid_X, valid_y),
        )


if __name__ == "__main__":
    # defining training, test and valid subjects
    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15]
    valid_subjects = [10]
    test_subjects = [12]
    # instantiate the training model process
    process = TrainModel(
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        valid_subjects=valid_subjects,
        n_rows=1,
        n_cols=2,
        n_timesteps=8,
        n_features=6,
        n_conv_filters=1024,
        n_dense_nodes=512,
        n_output_nodes=1,
        n_seq=4,
        batch_size=32,
        epochs=1,
    )
    # run training
    process.train()

    # extract test data
    test_X, test_y = process.load_data(subject_set="test")

    # get predictions
    yhat = process.model.predict(test_X)
    # get mae
    mae = mean_absolute_error(test_y, yhat)
    # plot predictions
    timesteps = list(range(len(yhat)))
    plt.plot(timesteps, test_y, "k", label="ground truth")
    plt.plot(timesteps, yhat, "-r", label="predictions")
    plt.legend()
    plt.show()
