from IPython.display import HTML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from time import time
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
import torch
from torch.utils.data import Dataset, TensorDataset, IterableDataset, DataLoader
import torch.nn as nn
import math
from utils import process_file, create_dataset
import torch.optim as optim
import torch.utils.data as data
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import pacf
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
