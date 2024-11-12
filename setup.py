# Data manipulation and scientific computing
import numpy as np
import pandas as pd
import polars as pl

# Machine learning and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Deep learning - TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Add, BatchNormalization, Concatenate, Dense, Dropout,
    GRU, Input, LSTM, Lambda, Multiply
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Visualization and display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
from scipy.stats import gaussian_kde

# Progress tracking
from tqdm import tqdm
from tqdm.notebook import tqdm

# File handling
import pickle
import os

import os

# Change this to the path of your repository
repo_path = '/content/nfl_motion_coverage_overlap'
os.chdir(repo_path)

# GitHub functions
from data_cleaning.data_cleaning import *
from data_cleaning.utils import *
from model.model_predictions import *
from model.model_train import *
