import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from IPython.display import HTML
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from time import time
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
import torch
from torch.utils.data import Dataset, TensorDataset, IterableDataset, DataLoader
import torch.nn as nn
import math
from sklearn.preprocessing import MaxAbsScaler


def process_file():
    df = pd.read_csv('./Datasets/recorridos-realizados-2018.csv', encoding='latin-1')
    # Convertir fecha_hora_retiro a datetime
    df.bici_Fecha_hora_retiro = df.bici_Fecha_hora_retiro.apply(lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M:%S'))
    df['date'] = df.bici_Fecha_hora_retiro.apply(lambda x: x.replace(minute=0, second=0))
    
    #limpieza de datos
    fecha_limite = pd.to_datetime('2016-08-01 00:00:00')
    df_shorten = df[df['date'] >= fecha_limite].copy()
    bicis_por_dia = df_shorten.groupby('date').bici_id_usuario.count().resample('D').sum()

    df_trimmed = df_shorten[df_shorten['date'] <= pd.to_datetime('2017-01-01 23:59:59')].copy()
    bicis_por_dia_trimmed = df_trimmed.groupby('date').bici_id_usuario.count().resample('D').sum()
    
    #df_trimmed = df_shorten[df_shorten['date'] <= pd.to_datetime('2017-12-01 23:59:59')].copy()
    #bicis_por_dia_trimmed = df_trimmed.groupby('date').bici_id_usuario.count().resample('D').sum()
    
    X = bicis_por_dia_trimmed
    X = pd.DataFrame(X)
    #X = X.reset_index()
    
    timeseries = X['bici_id_usuario'].values.astype('float32')
    timeseries = timeseries.reshape(-1, 1)
    return timeseries

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]  # Extract the feature window
        target = dataset[i+1:i+lookback+1]  # Extract the target window (shifted by 1 step)
        X.append(feature)
        y.append(target)
    # Convert lists to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    # Reshape y to add an extra dimension
    #y = y.unsqueeze(-1)
    #X = X.unsqueeze(-1)
    return X, y

class LogMaxAbsScaler:
    def __init__(self):
        self.scaler = MaxAbsScaler()

    def fit(self, X, y=None):
        self.scaler.fit(np.log(X+1))
        return self
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def transform(self, X):
        return self.scaler.transform(np.log(X+1))

    def inverse_transform(self, X):
        return np.exp(self.scaler.inverse_transform(X)) - 1

def scale_data(X_train, X_test, y_train, y_test, scaler_cls=MaxAbsScaler):
    # Scale X data
    scaler_x = scaler_cls()

    X_train_scaled = scaler_x.fit_transform(X_train.squeeze().numpy().reshape(-1, 1)).reshape(X_train.shape)
    X_test_scaled = scaler_x.transform(X_test.squeeze().numpy().reshape(-1, 1)).reshape(X_test.shape)

    # Scale y data
    scaler_y = scaler_cls()
    y_train_scaled = scaler_y.fit_transform(y_train.squeeze().numpy().reshape(-1, 1)).reshape(y_train.shape)
    y_test_scaled = scaler_y.transform(y_test.squeeze().numpy().reshape(-1, 1)).reshape(y_test.shape)

    # Convert scaled data to torch tensors
    X_train_scaled = torch.tensor(X_train_scaled)
    X_test_scaled = torch.tensor(X_test_scaled)
    y_train_scaled = torch.tensor(y_train_scaled)
    y_test_scaled = torch.tensor(y_test_scaled)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_x, scaler_y
