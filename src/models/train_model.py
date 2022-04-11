from __future__ import annotations
import optparse
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .define_model import (
    cnnLSTMModel,
    convLSTMModel,
    mlpModel,
    convModel,
)
from src.features.build_features import DataBlock, to_supervised, to_supervised_shuffled

from pickle import dump, load

np.random.seed(42)


def scale_and_encode(dataframe):
    """
    Function to scale numerical features and one hot encode categorical ones

    Args:
        dataframe: pd.DataFrame -> a pandas dataframe containing the data
    Returns:
        self.scaled_array:np.array -> a numpy array of scaled and encoded features
    """
    # the numeric features which are not dependent on the subject description
    numeric_features = ["bvp", "acc_x", "acc_y", "acc_z", "bmi", "age"]
    # cat_features = ["sport"]
    # create a pipeline to do the transformation
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    # categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder())])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            # ("cat", categorical_transformer, cat_features),
        ],
        remainder="passthrough",
    )
    # fit the columntransformer to the dataframe
    preprocessor.fit(dataframe)
    # save the preprocessor as we will fit this scaler to validation and testing sets
    dump(preprocessor, open("models/scaler_and_encoder.pkl", "wb"))
    # # return the transformed array
    return preprocessor.transform(dataframe)


class TrainModel:
    """
    Class to handle training using a convLSTM model
    """

    def __init__(
        self,
        train_subjects: list,
        valid_subjects: list,
        n_timesteps: int,
        n_features: int,
        n_conv_layers: int,
        n_conv_filters: int,
        kernel_size: int,
        n_lstm_units: int,
        n_dense_nodes: int,
        n_output_nodes: int,
        n_seq: int,
        batch_size: int,
        epochs: int,
        scaler_encoder=None,
    ):
        # define the model
        self.model = cnnLSTMModel(
            n_conv_layers=n_conv_layers,
            n_conv_filters=n_conv_filters,
            kernel_size=kernel_size,
            n_lstm_units=n_lstm_units,
            n_dense_nodes=n_dense_nodes,
            n_output_nodes=n_output_nodes,
            input_shape=(None, n_timesteps // n_seq, n_features),
        )
        # compile the model
        self.model.compile(loss="mse", metrics="mae", optimizer="adam")
        # define the train, test and valid subjects
        self.train_subjects = train_subjects
        self.test_subjects = []
        self.valid_subjects = valid_subjects
        # define the number of timesteps used in prediction
        self.timesteps = n_timesteps
        # define number of features used in the model
        self.features = n_features
        # # define the length of each subsequence
        self.seq = n_seq
        # define the batch size
        self.batch_size = batch_size
        # define epochs
        self.epochs = epochs
        # valid scores
        self.valid_score = 0
        # load scaler
        self.scaler_encoder = scaler_encoder

    def load_data(self, subject: int):
        """
        Function to load data for training

        Args:
            subject: int -> the subject for which data is being loaded

        Returns:
            X,y : np.array -> training data and labels
        """
        # load the dataframe
        data = DataBlock("S{}".format(subject), "data/raw/")
        df = data.raw_dataframe
        # name the columns
        df.columns = [
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
        # if scaling and encoding needs to be done, load the scaler encoder and transform the dataframe
        if self.scaler_encoder:
            df = self.scaler_encoder.transform(df)
            X, y = to_supervised(np.array(df), self.timesteps, 1)
        # reshape the X array to meet the requirements of the model
        X = self.reshape(X)
        return X, y

    def train(self):
        """
        Function to run training
        """
        for sub in self.train_subjects:
            # load training and validation data
            print("-------------------------------------")
            print("training on subject - {}".format(sub))
            print("-------------------------------------")
            train_X, train_y = self.load_data(subject=sub)

            # define callbacks
            # early stopping
            es_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)

            log_dir = "models/logs/fit/" + datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S"
            )
            # tensorboard callback
            tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
            # fit the model and save history
            self.model.fit(
                train_X,
                train_y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[es_callback, tb_callback],
            )
        print("-------------------------------------")
        print("testing on subject - {}".format(self.valid_subjects[0]))
        print("-------------------------------------")
        # check performance on hold out validation set
        valid_X, valid_y = self.load_data(subject=self.valid_subjects[0])
        yhat = process.model.predict(valid_X)
        # calculate mae of model predictions on validation data
        mae = mean_absolute_error(valid_y, yhat)
        self.valid_score = mae
        # save the model
        self.model.save("models/ckpoints/model_{}".format(self.valid_subjects[0]))

    # def train_shuffled(
    #     self,
    #     train_X: np.array,
    #     train_y: np.array,
    #     valid_X: np.array,
    #     valid_y: np.array,
    #     valid_subject: int,
    # ):
    #     """
    #     Function to run training
    #     """
    #     # define callbacks
    #     # early stopping
    #     es_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)
    #     log_dir = "models/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #     # tensorboard callback
    #     tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    #     # fit the model and save history
    #     self.model.fit(
    #         train_X,
    #         train_y,
    #         epochs=self.epochs,
    #         batch_size=self.batch_size,
    #         callbacks=[es_callback, tb_callback],
    #     )

    #     yhat = process.model.predict(valid_X)
    #     mae = mean_absolute_error(valid_y, yhat)

    #     self.valid_score = mae
    #     self.model.save("models/ckpoints/model_{}".format(valid_subject))

    def reshape(self, X: np.array):
        "Function which reshapes the input data into the required shape for CNN LSTM model"
        return X.reshape(
            (X.shape[0], self.seq, self.timesteps // self.seq, self.features)
        )


if __name__ == "__main__":

    total_subjects = list(range(1, 16))
    val_scores = []
    # iterate through each subject and treat it as validation set
    for i in total_subjects:
        print("******************************************")
        print("training fold - {}".format(i))
        print("******************************************")
        # defining training and validation subjects
        train_subjects = [x for x in total_subjects if x != i]
        valid_subjects = [i]
        # initiate a list of dataframes
        list_of_dfs = []
        # append all the dataframes in the training set
        for subject in train_subjects:
            data = DataBlock("S{}".format(subject), "data/raw/")
            df = data.raw_dataframe
            list_of_dfs.append(df)
        # create a concatenated dataframe
        frames = pd.concat(list_of_dfs)
        # scale and encode training set
        sf_frames = scale_and_encode(frames)
        # use the saved scaler encoder for later use with validation set
        saved_scaler_encoder = load(open("models/scaler_and_encoder.pkl", "rb"))
        # define number of features
        n_features = 8
        # instantiate the training model process -> for each training fold, the model is freshly initiated
        process = TrainModel(
            train_subjects=train_subjects,
            valid_subjects=valid_subjects,
            n_timesteps=8,
            n_features=n_features,
            n_conv_layers=2,
            n_conv_filters=20,
            kernel_size=4,
            n_lstm_units=64,
            n_dense_nodes=32,
            n_output_nodes=1,
            n_seq=1,
            batch_size=100,
            epochs=100,
            scaler_encoder=saved_scaler_encoder,
        )
        # run training
        process.train()
        # print and save validation scores
        print(
            "validation score on subject -{} ".format(valid_subjects[0]),
            process.valid_score,
        )
        val_scores.append(process.valid_score)

    print(val_scores)
