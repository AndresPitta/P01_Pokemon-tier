# authors: Andres Pitta
# date: 2020-02-23

'''This script will generate exploratory data analysis visualizations. It takes as arguments the file were the root 
file is, the path where the visualizations will be saved.

Usage: eda.py [--MODELS_FILE_PATH=<MODELS_FILE_PATH>] [--EDA_FILE_PATH=<EDA_FILE_PATH>]

Options:
--MODELS_FILE_PATH=<MODELS_FILE_PATH>  Path (including filename) to gather the .pic file. [default: results/models/models.pic]
--CHOSEN_MODEL=<CHOSEN_MODEL>  Name of the chosen model [default: 'OVO - logistic regression']
--CATEGORICAL_FEATURES=<CATEGORICAL_FEATURES>  String of categorical features separated by commas [default: Mega,Has_ST]
--NUMERICAL_FEATURES=<NUMERICAL_FEATURES>  String of numerical features separated by commas [default: HP,Attack,Defense,Special_attack,Special_defense,Speed]
'''

from docopt import docopt

# Modeling 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Pipeline

from sklearn.pipeline import Pipeline

# Preprocessing

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Data structures

import pandas as pd

# Plotting

import altair as alt
import plotly.graph_objects as go
from selenium import webdriver

# Other
import os
import sys

#browser = webdriver.Chrome('C:\webdrivers\chromedriver.exe')


opt = docopt(__doc__)

def main(data_file_path, eda_file_path):
    assert os.path.isfile(data_file_path), "File does not exist"
    assert os.path.isdir(eda_file_path), "EDA_FILE_PATH does not exist, please create a 'figures' folder in results"

    data = pd.read_csv(data_file_path)
    
    make_correlation(data, eda_file_path)
    make_bars(data, eda_file_path)

def choose_model(model_name, results):
    """
    Returns the model selected by the user

    Parameters:
    model_name -- (string) name of the selected model
    results -- (dictionary) dictionary containing model, train error, validation error
    and elapsed training and validation time
   
    Returns:
    model -- scikit learn model
    """
    model = results[model_name][0]
    return model

def hyperparameter_choosing():

if __name__ == "__main__":
     main(opt["--TRAIN_FILE_PATH"], opt["--TEST_FILE_PATH"])