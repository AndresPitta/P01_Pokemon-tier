# authors: Andres Pitta
# date: 2020-02-23

'''This script evaluates a set of models a prints the results so that the user chooses the model.

Usage: modeling.py [--TRAIN_FILE_PATH=<TRAIN_FILE_PATH>] [--TEST_FILE_PATH=<TEST_FILE_PATH>] [--RESULTS_FILE_PATH=<RESULTS_FILE_PATH>] [--IMPORTANCES_FILE_PATH=<IMPORTANCES_FILE_PATH>] [--MODEL_DUMP_PATH=<MODEL_DUMP_PATH>] [--IMPORTANCE_PLOT_PATH=<IMPORTANCE_PLOT_PATH>] [--CATEGORICAL_FEATURES=<CATEGORICAL_FEATURES>] [--NUMERICAL_FEATURES=<NUMERICAL_FEATURES>] [--RESULTS_FINAL_PATH=<RESULTS_FINAL_PATH>]

Options:
--TRAIN_FILE_PATH=<TRAIN_FILE_PATH>  Path (including filename) to gather the csv file. [default: data/pokemon_smogon_competitive_train.csv]
--TEST_FILE_PATH=<TEST_FILE_PATH>  Path (including filename) to gather the test csv file. [default: data/pokemon_smogon_competitive_test.csv]
--RESULTS_FILE_PATH=<RESULTS_FILE_PATH>  Path to output Results table. [default: results/pokemon_models.csv]
--IMPORTANCES_FILE_PATH=<IMPORTANCES_FILE_PATH>  Path to output Feature Importance table. [default: results/pokemon_feature_importances.csv]
--MODEL_DUMP_PATH=<MODEL_DUMP_PATH>  Path to output Models table. [default: results/models/final_model.pic]
--IMPORTANCE_PLOT_PATH=<IMPORTANCE_PLOT_PATH>  Path to print importance plot. [default: results/figures/importance_plot.png]
--CATEGORICAL_FEATURES=<CATEGORICAL_FEATURES>  String of categorical features separated by commas [default: Type1,Type2,Mega,Has_ST]
--NUMERICAL_FEATURES=<NUMERICAL_FEATURES>  String of numerical features separated by commas [default: HP,Attack,Defense,Special_attack,Special_defense,Speed]
--RESULTS_FINAL_PATH=<RESULTS_FINAL_PATH>  Path to output Final model's Results table. [default: results/pokemon_final_model.csv]
'''

from docopt import docopt

# Modeling 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Pipeline

from sklearn.pipeline import Pipeline

# Preprocessing

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Data structures

import pandas as pd

# Plotting

import altair as alt
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

# Other
import os
import sys
import pickle
import time

#driver = webdriver.Chrome(ChromeDriverManager().install())
browser = webdriver.Chrome('C:\webdrivers\chromedriver.exe')
opt = docopt(__doc__)

def main(data_file_path, test_file_path, results_file_path, importances_file_path, model_dump_path, importance_plot_path, 
         cat_features, num_features, final_path):
    assert os.path.isfile(data_file_path), "TRAIN_FILE_PATH does not exist"
    assert os.path.isfile(test_file_path), "TEST_FILE_PATH does not exist"
    
    print("Model evaluation - Starting \n")
    data = pd.read_csv(data_file_path)
    test_data = pd.read_csv(test_file_path)
    
    categorical_features = cat_features.split(",")
    numeric_features = num_features.split(",")
    all_features = categorical_features + numeric_features
    
    print(f"Used features: {all_features} \n")
    
    X = data[all_features]
    y = data['Tier_2']
    X_test = test_data[all_features]
    y_test = test_data['Tier_2']
    
    print("Splitting the data... \n", end='')
    X_valid, X_train, y_valid, y_train = data_splitting(X, y, 0.7)
    
    print("Setting the preprocessor... \n", end='')
    preprocessor = preprocessing(categorical_features, numeric_features)
    
    models = {
          'decision tree': DecisionTreeClassifier(),
          'kNN': KNeighborsClassifier(),
          'OVR - logistic regression': OneVsRestClassifier(LogisticRegression(solver ='lbfgs')),
          'OVR - RBF SVM' :  OneVsRestClassifier(SVC(gamma = 'scale')), 
          'OVO - logistic regression': OneVsOneClassifier(LogisticRegression(solver ='lbfgs')),
          'OVO - RBF SVM' :  OneVsOneClassifier(SVC(gamma = 'scale')), 
          'random forest' : RandomForestClassifier(), 
          'xgboost' : XGBClassifier(),
          'lgbm': LGBMClassifier(),
          'Dummy': DummyClassifier()
         }
    
    print("Evaluating models... \n", end='')
    results, importances = evaluate_model(X_train, y_train, X_valid, 
                                          y_valid, preprocessor, models,
                                          categorical_features, numeric_features)
    
    # Turning the dictionary into a df
    
    results_df = pd.DataFrame(results).T
    results_df.columns = ["Model", 
                          "Train Accuracy", 
                          "Validation Accuracy", 
                          "Time in seconds"]
    
    
    print("Printing results... \n", end='')
    results_df.to_csv(results_file_path)
    
    print("Printing importances... \n", end='')
    importances.to_csv(importances_file_path)
    
    print("Saving plots... \n", end='')
    plot_feature_importance(importances, importance_plot_path)
    print(f"Importance plot saved in {importance_plot_path}")
    
    chosen_model = 'OVR - logistic regression'
    print(f"The chosen model is {chosen_model}\n", end='')
    final_model = results[chosen_model][0]
    test_results = test_model(X_train, y_train, X_test, y_test, preprocessor, final_model, chosen_model)
    
    print("Dumping models... \n", end='')
    dump_model(model_dump_path, final_model)
    
    print("Printing results - final model... \n", end='')
    test_results.to_csv(final_path)
    
    print("Model evaluation - Finished")

def preprocessing(categorical_features, numeric_features):
    """
    Creates preprocessing step of the pipeline using one hot encoding representations
    
    Parameters:
    categorical_features -- (list) list of categorical features to preprocess
    categorical_features -- (list) list of numerical features to preprocess
    
    Returns: 
    preprocessor -- (sklearn.compose.ColumnTransformer) one hot encoding preprocessor
    """

    categorical_transformer = Pipeline(steps=[
                                          ('imputer', SimpleImputer(strategy='constant', 
                                                                    fill_value='no type')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                         ])

    numerical_transformer = Pipeline(steps=[
                                              ('standard', StandardScaler())
                                             ])

    preprocessor = ColumnTransformer(
                                     transformers=[
                                        ('cat', categorical_transformer, categorical_features),
                                        ('num', numerical_transformer, numeric_features)
                                    ])

    return preprocessor

def data_splitting(X, y, train_test_size):
    """
    Splits the data into a validation and a train set
    
    Parameters:
    X -- (dataframe) Explanatory variables matrix
    y -- (series) Response variable series
    
    Returns:
    X_valid -- (dataframe) validation X
    X_train -- (dataframe) train X
    y_valid -- (series) validation y
    y_train -- (series) train y
    """

    X_valid, X_train, y_valid, y_train = train_test_split(X, y, train_size=train_test_size, random_state = 1234)
    
    return X_valid, X_train, y_valid, y_train


def evaluate_model(X_train, y_train, X_valid, y_valid, preprocessor, models, categorical_features, numeric_features):
    """
    Evaluates a group of models

    Parameters:
    X_valid -- (dataframe) validation X
    X_train -- (dataframe) train X
    y_valid -- (series) validation y
    y_train -- (series) train y
    preprocessor -- (sklearn.compose.ColumnTransformer) one hot encoding preprocessor
    models -- (dictionary) models dictionary
    
    Returns:
    results -- (dictionary) dictionary containing model, train error, validation error
    and elapsed training and validation time
    
    importances -- (dataframe) feature importances
    """
    results = {}
    
    # Retrieving the names of the variables just once
    
    preprocessor.fit(X_train)
    
    categorical_names = pd.DataFrame(preprocessor.named_transformers_['cat']\
                                                 .named_steps['onehot']\
                                                 .get_feature_names(categorical_features), 
                                     columns = ['Features'])
    
    numerical_names = pd.DataFrame(numeric_features, columns = ['Features'])
    
    importance_df = pd.concat([categorical_names, numerical_names], axis = 0)\
                      .reset_index()\
                      .drop(columns = ['index'])
    
    # List of models without feature_importances_
    
    models_fi = ['kNN', 'OVR - logistic regression', 'OVR - RBF SVM', 
                 'OVO - logistic regression', 'OVO - RBF SVM', 'Dummy',
                 'lgbm']
        
    for model_name, model in models.items():
        
        # Timing the model
        
        t = time.time()

        # Fitting the model as a pipeline
        
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        clf.fit(X_train, y_train);
        tr_acc, valid_acc = clf.score(X_train, y_train), clf.score(X_valid, y_valid)
        
        elapsed_time = time.time() - t
        
        results[model_name] = [clf, round(tr_acc,3), round(valid_acc,3), round(elapsed_time,4)]
       
        # Not evaluating importances for those models in which it cannot be evaluated
        
        if model_name not in models_fi:
            importances = pd.DataFrame(model.feature_importances_, 
                                       columns = [model_name])
            importance_df = pd.concat([importance_df, importances], 
                                      axis = 1)
    
    return results, importance_df

def dump_model(model_dump_path, model):
    """
    Dumps a model in a specified path
    
    Parameters:
    model_dump_path -- (string) path to dump the model
    model -- (model object) model to dump
    """
    with open(model_dump_path, 'wb') as fp:
        pickle.dump(model, fp)
        
def plot_feature_importance(data, importance_plot_path):
    """
    Plots an importance plot
    
    Parameters:
    data -- (dataframe) Importance dataframe
    importance_plot_path -- (string) path to print the plot
    """
    data_melted = data.melt(id_vars = ['Features'],
                            var_name = 'Model',
                            value_name = 'Importance')
    
    chart = alt.Chart(data_melted).mark_bar().encode(
            alt.X('Importance:Q'),
            alt.Y('Features:O', 
                  sort = alt.EncodingSortField(field = 'Importance', order = 'descending'),
                  axis = alt.Axis(title = 'Features', tickCount = 8)),
            alt.Color('Model:N'),
            alt.Row('Model:N')
    ).properties(width=500, 
                 height=1000, 
                 title = f'Feature Importance using Different Models'
    ).configure_axis(labelFontSize=15, 
                     titleFontSize=22
    ).configure_title(
        fontSize=26
    )
        
    chart.save(importance_plot_path)
    
def test_model(X, y, X_test, y_test, preprocessor, final_model, model_name):
    """
    Evaluates the final model in a test dataset
    
    Parameters:
    X -- (dataframe) train X
    y -- (dataframe) train y
    X_test -- (dataframe) test X
    y_test -- (dataframe) test y
    preprocessor -- (sklearn.compose.ColumnTransformer) one hot encoding preprocessor
    final_model -- (model) final model
    model_name -- (string) Name of the model
    
    Returns: 
    results -- (dataframe) Dataframe with testing results
    """
    results = {
        'Model': model_name,
        'Train accuracy': final_model.score(X, y),
        'Test accuracy': final_model.score(X_test, y_test)
    }
    
    results_df = pd.DataFrame(results, index = [0])
    
    return results_df
    

if __name__ == "__main__":
     main(opt["--TRAIN_FILE_PATH"], opt["--TEST_FILE_PATH"], opt["--RESULTS_FILE_PATH"], 
          opt["--IMPORTANCES_FILE_PATH"], opt["--MODEL_DUMP_PATH"], opt["--IMPORTANCE_PLOT_PATH"],
          opt["--CATEGORICAL_FEATURES"], opt["--NUMERICAL_FEATURES"], opt["--RESULTS_FINAL_PATH"])