'''


    @Description: This script performs a hyperparameter tuning for an LSTM autoencoder model
    on nuclear reactor data. It uses Keras Tuner to optimize the model architecture and training parameters.
    It loads training and validation datasets, normalizes them, and then builds and tunes the model based on validation loss.


    USAGE
    Copy and paste the following command in a cmd environment
    Make sure you have installed all the appropriate libaries
    I ran the code in a conda environment where I have installed tensorflow with GPU

    python reactor_anomaly_detection_training.py

'''

from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras import optimizers
from keras.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import datetime
from keras.models import load_model
import keras_tuner as kt
import os

# Configuration dictionary
config = {
    "number_of_epochs": 20,
    "batch_size": 32,
    "model_name": "./models/lstm_autoencoder_replay_.model",
    "save_models": True,  # False
    "training_folder": './Full cycles and Startups/Power_Cycle/training/',
    "validation_folder": './Full cycles and Startups/Power_Cycle/validation/',
    "test_folder": './Full cycles and Startups/Power_Cycle/testing/'
}


def find_null_values(df):
    """
    Removes rows with null values from the DataFrame.
    """
    df = pd.DataFrame(df)
    df.dropna(inplace=True)

    return df


def load_data(filename, cols_to_be_read, percentage):
    '''
        This function loads the data from a .csv file to a pandas dataframe.
        Percentage parameter defines the number of rows to be loaded
    '''

    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    df = df.loc[:, cols_to_be_read]

    length = len(df)

    try:
        if 0 < percentage <= 1.0:

            number_of_rows = int(percentage*length)
            df = df[:number_of_rows]

            return df
    except:
        raise ValueError("values in percentage should be in the range (0, 1]")


def split_data(df, train_size=0.8):
    """
    Splits the DataFrame into training and validation sets.
    """
    split_index = int(len(df) * train_size)
    return df[:split_index], df[split_index:]


def normalize_data(train_df, val_df, normalization_mode, normalization_model_name):
    """
    Normalizes the training and validation data using MinMax or Standard scaling.
    """
    # Normalize data
    if normalization_mode == 'MinMax':
        scaler = MinMaxScaler()
    elif normalization_mode == 'Standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Provide MinMax or Standard normalization methods.")

    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    # test_df_scaled = scaler.transform(test_df)
    # Export the Normalized model to be used later
    if config['save_models'] == True:
        try:
            joblib.dump(scaler, './models/' + normalization_model_name)
            print("[INFO] Normalization model saved.")
        except:
            print('[Error] Normalization model could not be saved.')

    return pd.DataFrame(train_scaled), pd.DataFrame(val_scaled)


def min_max_normalizer(df, feature_list, min_max_csv):
    """
    Normalize selected features in a DataFrame using Min-Max scaling,
    based on precomputed min-max values from a CSV file.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the features to be normalized.
        feature_list (list): A list of features to normalize.
        min_max_csv: Path to the CSV file containing the min and max values for each feature.

    Returns:
        pd.DataFrame: A DataFrame with the same structure as `df`, but with normalized values for selected features.
    """
    # Load the min-max values from CSV
    min_max_df = pd.read_csv(min_max_csv)

    # Convert to dictionary for fast lookup
    min_vals = dict(zip(min_max_df["Feature"], min_max_df["Min"]))
    max_vals = dict(zip(min_max_df["Feature"], min_max_df["Max"]))

    # Create a copy of the original DataFrame
    normalized_df = df.copy()

    # Apply Min-Max Normalization to selected features
    for feature in feature_list:
        # print(feature)
        if feature in min_vals and feature in max_vals:
            min_val = min_vals[feature]
            max_val = max_vals[feature]

            # Avoid division by zero
            if max_val != min_val:
                normalized_df[feature] = (
                    df[feature] - min_val) / (max_val - min_val)
            else:
                normalized_df[feature] = 0  # Assign 0 if no range

    return normalized_df[feature_list]


def to_sequences(data, seq_size=10):
    """
    Converts a DataFrame into sequences for LSTM input.
    """
    x_values = []

    for i in range(len(data)-seq_size):
        # print(i)
        x_values.append(data.iloc[i:(i+seq_size)].values)

    return np.array(x_values)


def to_sequences_(x, seq_size):
    '''
        Creates appropriate sequences from the data by checking if the index is concecutive
    '''
    x_values = []
    index_values = x["Unnamed: 0"]

    for i in range(len(x)-seq_size):
        # print(i)
        seq = x.iloc[i:(i + seq_size)]
        seq_index = list(index_values[i:(i + seq_size)])

        # Check if all indices in seq_index are successive
        is_continuous = True

        for j in range(1, len(seq_index)):
            if seq_index[j] - seq_index[j - 1] != 1:
                is_continuous = False
                break

        if is_continuous:
            # only append into list if all time values are continuous
            # Append only the data, excluding the index
            x_values.append(seq.drop(columns=["Unnamed: 0"]).values)

    return np.array(x_values)


def plot_metric(history, metric='loss'):
    """
    Plots training and validation metrics.
    """
    plt.plot(history.history[metric], label='Training')
    plt.plot(history.history[f'val_{metric}'], label='Validation')
    plt.title(f'{metric.capitalize()} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()


def build_model(hp, trainX):
    '''
        Builds an LSTM Autoencoder model with hyperparameter tuning.
    '''
    # Define the model architecture
    model = Sequential()

    model.add(LSTM(units=hp.Choice('units_1', [64, 128, 256]),
                   return_sequences=True,
                   activation='sigmoid',
                   input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dropout(hp.Choice('dropout_1', [0.1, 0.2, 0.3])))

    # Hidden LSTM layers
    for i in range(hp.Int('num_hidden_layers', 1, 3)):
        model.add(LSTM(
            units=hp.Choice(f'units_{i}', [32, 64, 128]),
            activation='sigmoid',
            return_sequences=True
        ))

        # Bottleneck layer
    model.add(LSTM(
        units=hp.Choice('units_bottleneck', [16, 32, 64]),
        activation='sigmoid',
        return_sequences=False
    ))

    model.add(RepeatVector(trainX.shape[1]))

    # Decoder
    model.add(LSTM(
        units=hp.Choice('units_decoder', [32, 64, 128]),
        activation='sigmoid',
        return_sequences=True
    ))

    model.add(Dropout(hp.Float('dropout_decoder', 0.1, 0.5, step=0.1)))

    model.add(LSTM(128, activation='sigmoid', return_sequences=True))

    model.add(TimeDistributed(Dense(trainX.shape[2])))

    model.compile(
        optimizer=optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='mse'
    )


def main():
    print("...START TRAINING.....")
    start_time = datetime.datetime.now()
    print(f"[INFO] Training started at: {start_time}")

    # Load training data

    feature_list = ["Unnamed: 0", "nfd-1-cps", "nfd-3-pwr", "nfd-4-flux", "rr-active-state",
                    "rr-position", "ss1-active-state", "ss1-position", "ss2-active-state", "ss2-position"]

    min_max_csv = './global_feature_min_max_summary.csv'

    train_sequences = []  # List to store sequences from all files
    # load training data
    for file_name in os.listdir(config['training_folder']):
        if file_name.endswith('.csv'):
            file_path = os.path.join(config['training_folder'], file_name)
            df = pd.read_csv(file_path, usecols=feature_list)

            df.dropna(inplace=True)

            # Normalize the data
            df_norm = df.copy()
            df_norm.loc[:, feature_list] = min_max_normalizer(
                df, feature_list, min_max_csv=min_max_csv)

            df[feature_list] = df_norm[feature_list]

            # Prepare sequences for LSTM input
            sequences = to_sequences_(df, seq_size=10)

            train_sequences.append(sequences)

    # Stack all sequences into one tensor
    trainX = np.vstack(train_sequences)

    print('Train X shape:', trainX.shape)

    # load validation data
    val_sequences = []  # List to store sequences from all files
    for file_name in os.listdir(config['validation_folder']):
        if file_name.endswith('.csv'):
            file_path = os.path.join(config['validation_folder'], file_name)
            df = pd.read_csv(file_path, usecols=feature_list)

            # Replace NaN values with 0
            # df.fillna(0, inplace=True)
            df.dropna(inplace=True)
            df_norm = df.copy()

            # Normalize the data
            df_norm.loc[:, feature_list] = min_max_normalizer(
                df, feature_list, min_max_csv=min_max_csv)

            df[feature_list] = df_norm[feature_list]

            # Prepare sequences for LSTM input
            sequences = to_sequences_(df, seq_size=10)

            val_sequences.append(sequences)
    # Stack all sequences into one tensor
    valX = np.vstack(val_sequences)
    print('Validation X shape:', valX.shape)

    def build_model(hp):
        '''
            Builds an LSTM Autoencoder model with hyperparameter tuning.
        '''
        # Define the model architecture
        model = Sequential()

        model.add(LSTM(units=hp.Choice('units_1', [64, 128, 256]),
                       return_sequences=True,
                       activation='sigmoid',
                       input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dropout(hp.Choice('dropout_1', [0.1, 0.2, 0.3])))

        # Hidden LSTM layers
        for i in range(hp.Int('num_hidden_layers', 1, 3)):
            model.add(LSTM(
                units=hp.Choice(f'units_{i}', [32, 64, 128]),
                activation='sigmoid',
                return_sequences=True
            ))

            # Bottleneck layer
        model.add(LSTM(
            units=hp.Choice('units_bottleneck', [16, 32, 64]),
            activation='sigmoid',
            return_sequences=False
        ))

        model.add(RepeatVector(trainX.shape[1]))

        # Decoder
        model.add(LSTM(
            units=hp.Choice('units_decoder', [32, 64, 128]),
            activation='sigmoid',
            return_sequences=True
        ))

        model.add(Dropout(hp.Float('dropout_decoder', 0.1, 0.5, step=0.1)))

        model.add(LSTM(hp.Choice('units_final_decoder', [64, 128, 256]),
                       activation='sigmoid', return_sequences=True))

        model.add(TimeDistributed(Dense(trainX.shape[2])))

        print(model.summary())

        model.compile(
            optimizer=optimizers.Adam(
                hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
            loss='mse'
        )

        return model

    # Initialize the hyperparameter tuner
    print("[INFO] Initializing hyperparameter tuner...")

    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=20,
        directory='tuner_logs',
        # callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5)],
        project_name='autoencoder_architecture_tuning'
    )

    tuner.search(trainX, trainX,
                 epochs=30,
                 validation_data=(valX, valX),
                 batch_size=32)

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best hyperparameters:")
    for key in best_hps.values:
        print(f"{key}: {best_hps.get(key)}")

    end_time = datetime.datetime.now()
    print(f"[INFO] Training completed in: {end_time - start_time}")


if __name__ == "__main__":
    main()
