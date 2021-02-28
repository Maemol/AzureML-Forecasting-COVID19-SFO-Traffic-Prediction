import argparse
import os

from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Dataset, Experiment, Run

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
import tensorflow.keras as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Arguments definition
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, dest='dataset', help='dataset')
parser.add_argument('--model_type', type=str, dest='model_type', help='model_type')

parser.add_argument('--n_epochs', type=int, dest='n_epochs', default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, dest='batch_size', default=64, help='batch_size')
parser.add_argument('--n_layers', type=int, dest='n_layers', default=1, help='number of layers')
parser.add_argument('--n_neurons', type=int, dest='n_neurons', default=100, help='number of neurons')
parser.add_argument('--look_back', type=int, dest='look_back', default=12, help='look_back')
parser.add_argument('--dropout', type=float, dest='dropout', default=0.0, help='dropout')


args = parser.parse_args()
dataset = args.dataset
model_type = args.model_type

n_epochs = args.n_epochs
batch_size = args.batch_size
n_layers = args.n_layers
n_neurons = args.n_neurons
look_back = args.look_back
dropout = args.dropout


run = Run.get_context()

workspace = run.experiment.workspace


# Get a dataset by name
dataset_prepared = Dataset.get_by_name(workspace=workspace, name=dataset, version='latest')

# Load a TabularDataset into pandas DataFrame
df = dataset_prepared.to_pandas_dataframe()

# Data preparation
df = df.set_index('date', inplace=False)


# Functions

def dnn_2d(df, look_back):
    X,Y =[], []
    for i in range(len(df)-look_back):
        d=i+look_back
        X.append(df[i:d,0])
        Y.append(df[d,0])
    return np.array(X),np.array(Y)

def lstm_3d(df, look_back):
    X, Y =[], []
    for i in range(len(df)-look_back):
        d=i+look_back
        X.append(df[i:d,])
        Y.append(df[d,])
    return np.array(X), np.array(Y)


def model_loss(history):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    sns.despine(top=True)
    plt.show()
    run.log_image('Loss per epochs', plot=plt)

    return

def prediction_plot(y_test, test_predict,look_back):
    len_prediction=df.index[look_back:]
    plt.figure(figsize=(8,4))
    plt.plot(len_prediction, y_test[:], marker='.', label="actual")    
    plt.plot(len_prediction, test_predict[:], 'r', label="prediction")
    plt.tight_layout()
    sns.despine(top=True)
    plt.ylabel('Passenger Count', size=12)
    plt.xlabel('Time step', size=12)
    plt.legend(fontsize=12)    
    plt.show()
    run.log_image('Actual vs Predicted', plot=plt)

    return


class LogRunMetrics(Callback):
        # callback at the end of every epoch
        def on_epoch_end(self, epoch, log):
            # log a value repeated which creates a list
            run.log('Loss', log['loss'])
            run.log('val_loss', log['val_loss'])

# Variable

train_size = 160

    
# Model

# Simple DNN

def model_dnn(look_back, n_layers, n_neurons):
    model = Sequential()
    model.add(Dense(units=n_neurons, input_dim=look_back, activation='relu'))
    for i in range(n_layers):
        model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mse','mae'])
    return model

# LSTM

def model_lstm(look_back, n_layers, n_neurons, dropout):
    model=Sequential()    
    model.add(LSTM(n_neurons, activation='relu', input_shape=(1,look_back), dropout=dropout))
    for i in range(n_layers):
        model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(1))    
    model.compile(loss='mean_squared_error',  optimizer='adam',metrics = ['mse', 'mae'])    
    return model



if model_type == 'DNN':
    train, test = df.values[0:train_size,:], df.values[train_size:,:]

    # Training DNN
    model = model_dnn(look_back, n_layers, n_neurons)
    
    X_train,y_train = dnn_2d(train,look_back)
    X_test,y_test = dnn_2d(test,look_back)

    history = model.fit([X_train], [y_train], epochs = n_epochs, batch_size=batch_size, verbose=0,
                        validation_data=([X_test],[y_test]),
                        shuffle=False, callbacks=[LogRunMetrics()])
    
    train_score = model.evaluate(X_train, y_train, verbose=0)
    
    print('Train Root Mean Squared Error(RMSE): %.2f; Train Mean Absolute Error(MAE) : %.2f ' 
    % (np.sqrt(train_score[1]), train_score[2]))
    test_score = model.evaluate(X_test, y_test, verbose=0)
    print('Val Root Mean Squared Error(RMSE): %.2f; Test Mean Absolute Error(MAE) : %.2f ' 
    % (np.sqrt(test_score[1]), test_score[2]))
    
    run.log('MAE_train', train_score[2])
    run.log('MAE_val', test_score[2])
    run.log('RMSE_train', np.sqrt(train_score[1]))
    run.log('RMSE_val', np.sqrt(test_score[1]))
    run.tag("n_epochs", n_epochs)
    run.tag("batch_size", batch_size)
    run.tag("n_neurons", n_neurons)
    run.tag("n_layers", n_layers)
    run.tag("model_type", model_type)
    run.tag("look_back", look_back)
    run.tag("dropout", dropout)


    # Plot the loss
    model_loss(history)
    
    # Plot the actual vs prediction
    X,y = dnn_2d(df.values,look_back)
    test_predict = model.predict(X)
    prediction_plot(y, test_predict,look_back)
    
    # Model persistence 
    os.makedirs('outputs', exist_ok=True)
    model.save('outputs/'+str(run.id))
    

if model_type == 'LSTM':
    #create numpy.ndarray
    df_arr= df.values
    df_arr = np.reshape(df_arr, (-1, 1)) #LTSM requires more input features compared to RNN or DNN
    scaler = MinMaxScaler(feature_range=(0, 1))#LTSM is senstive to the scale of features
    df_arr = scaler.fit_transform(df_arr)
    
    test_size = len(df_arr)-train_size
    train, test = df_arr[0:train_size,:], df_arr[train_size:len(df_arr),:]
    trainX, trainY = lstm_3d(train, look_back)
    testX, testY = lstm_3d(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    model=model_lstm(look_back, n_layers, n_neurons, dropout)
    history = model.fit(trainX, trainY, epochs=n_epochs, batch_size=batch_size, validation_data=(testX, testY),
                        verbose=0, shuffle=False, callbacks=[LogRunMetrics()])

    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)

    # invert predictions

    train_predict = scaler.inverse_transform(train_predict)
    trainY = scaler.inverse_transform(trainY)
    test_predict = scaler.inverse_transform(test_predict)
    testY = scaler.inverse_transform(testY)

    run.log('MAE_train', mean_absolute_error(trainY[:,0], train_predict[:,0]))
    run.log('MAE_val', mean_absolute_error(testY[:,0], test_predict[:,0]))
    run.log('RMSE_train', np.sqrt(mean_squared_error(trainY[:,0], train_predict[:,0])))
    run.log('RMSE_val', np.sqrt(mean_squared_error(trainY[:,0], train_predict[:,0])))
    run.tag("n_epochs", n_epochs)
    run.tag("batch_size", batch_size)
    run.tag("n_neurons", n_neurons)
    run.tag("n_layers", n_layers)
    run.tag("model_type", model_type)
    run.tag("look_back", look_back)
    run.tag("dropout", dropout)
    
    # Plot the loss
    model_loss(history)
    
    # Plot the actual vs prediction
    X,y = lstm_3d(df_arr,look_back)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    test_predict = model.predict(X)
    y = scaler.inverse_transform(y)
    test_predict = scaler.inverse_transform(test_predict)    
    prediction_plot(y, test_predict,look_back)
    
    # Model persistence 
    os.makedirs('outputs', exist_ok=True)
    model.save('outputs/'+str(run.id))

run.complete()