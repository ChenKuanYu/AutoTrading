import os
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

raw_columns = ["open", "high", "low", "close"]

testing = 0
model_refresh_freq = 30

def load_data(filename):
    print("Load: {}".format(filename))
    data_df = pd.read_csv(filename, names=raw_columns, header=None)
    return data_df

def parse_tomorrow_up(data):
    output_df = data.copy()
    tomorrow_diff = []
    for i in range(output_df.shape[0]-1):
        tomorrow_diff.append(output_df.iloc[i+1, 3] - output_df.iloc[i+1,0])
    output_df["tomorrow_diff"] = pd.DataFrame(tomorrow_diff)
    output_df["tomorrow_ups"] = output_df["tomorrow_diff"].apply(lambda x: 1 if x >= 0 else 0)
    return output_df

def normalize_data(X_data):
    scaler = MinMaxScaler(feature_range=(0,1))
    X_data = np.log(X_data)
    return scaler.fit_transform(X_data)

def generate_lstm_X_y(X, y):
    package_size = 10
    X_lstm = []
    y_lstm = []
    for i in range(package_size,len(X)):
        X_lstm.append(X[i-package_size:i, :]) 
        y_lstm.append(y[i])
    return np.array(X_lstm), np.array(y_lstm)

def split_data(data):
    split_point = int(len(data)*0.8)
    clean_data = data.dropna()
    X = clean_data.iloc[:,0:4]
    y = clean_data.iloc[:,5]
    scaled_X = normalize_data(X)
    X, y = generate_lstm_X_y(scaled_X, y)
    X_train = X[:split_point, :, :]
    X_test = X[split_point:, :, :]
    y_train = y[:split_point]
    y_test = y[split_point:]
    return X_train, y_train, X_test, y_test

class Trader:
    def __init__(self):
        self.hold_stock = 0
        self.model = None
        self.train_df = None
        self.new_added_num = 0
    
    def buy_stock(self):
        if self.hold_stock == -1:
            self.hold_stock = 0
            return "1"
        elif self.hold_stock == 0:
            self.hold_stock = 1
            return "1"
        else:
            self.hold_stock = 1
            return "0"

    def sell_stock(self):
        if self.hold_stock == -1:
            self.hold_stock = -1
            return "0"
        elif self.hold_stock == 0:
            self.hold_stock = -1
            return "-1"
        else:
            self.hold_stock = 0
            return "-1"

    def create_model(self, input_shape):
        regressor = Sequential()
        if testing == 1:
            regressor.add(LSTM(units = 100, input_shape = input_shape))
        else:
            regressor.add(LSTM(units = 100, return_sequences=True, input_shape = input_shape))
            regressor.add(Dropout(0.2))
            regressor.add(LSTM(units=100))
            regressor.add(Dropout(0.2))
        regressor.add(Dense(units = 1, activation='sigmoid'))
        regressor.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
        print(regressor.summary())
        return regressor

    def train(self, data):
        print("Start to train: {}".format(data.describe()))
        self.train_df = data
        new_data_df = parse_tomorrow_up(data)
        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm = split_data(new_data_df)
        self.model = self.create_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
        self.model.fit(X_train_lstm, y_train_lstm, epochs = 100, batch_size = 16, validation_data=(X_test_lstm, y_test_lstm))
    
    def predict_action(self, entry):
        if self.train_df is None:
            print("Please train the model first")
            os._exit(0)
        print("Predict the item: {}".format(entry))
        print("Original data length: {}".format(self.train_df.shape))
        predict_range = 10
        full_df = pd.concat([self.train_df, entry], ignore_index=True)
        new_full_df = parse_tomorrow_up(full_df)
        X_verify = new_full_df.iloc[:,0:4]
        y_verify = new_full_df.iloc[:,5]
        scaled_X_verify = normalize_data(X_verify)
        lstm_X, lstm_y = generate_lstm_X_y(scaled_X_verify, y_verify)
        X_ver = lstm_X[-predict_range:]
        y_ver = lstm_y[-predict_range:]
        predicted_y = self.model.predict(X_ver)
        scaler = MinMaxScaler(feature_range=(0,1))
        scalered_predict = scaler.fit_transform(predicted_y)
        #print(scalered_predict)
        if(scalered_predict[-1] >= 0.5):
            return self.buy_stock()
        else:
            return self.sell_stock()

    def re_training(self, entry):
        print("Keep learn new data: {}".format(entry))
        self.train_df = pd.concat([self.train_df, entry], ignore_index=True)
        self.new_added_num += 1
        if self.new_added_num == model_refresh_freq:
            self.train(self.train_df)
            self.new_added_num = 0

if __name__ == '__main__':
    # You should not modify this part.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
            default='training_data.csv',
            help='input training data file name')
    parser.add_argument('--testing',
            default='testing_data.csv',
            help='input testing data file name')
    parser.add_argument('--output',
            default='output.csv',
            help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    training_data = load_data(args.training)
    trader = Trader()
    trader.train(training_data)

    testing_data = load_data(args.testing)
    with open(args.output, 'w') as output_file:
        for i in range(testing_data.shape[0]):
            row_df = testing_data.loc[i].to_frame().transpose()
            # We will perform your action as the open price in the next day.
            action = trader.predict_action(row_df)
            output_file.write("{}\n".format(action))
            # this is your option, you can leave it empty.
            trader.re_training(row_df)
            