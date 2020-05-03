# author: Andres Pitta
# date: 2020-02-23

'''This script wrangles and splits the data for ML purposes. It takes the following arguments as inputs: 
    the path were the root file is,
    the path where the test and train dataset are going to be saved, 
    the train/test set split in decimal numbers
    a boolean variable indicating if the dataset contains the new generation (unlabeled data)
    and a path to print the wrangled new gen dataset

Usage: wrangling.py [--DATA_FILE_PATH=<DATA_FILE_PATH>] [--TRAIN_FILE_PATH=<TRAIN_FILE_PATH>] [--TEST_FILE_PATH=<TEST_FILE_PATH>] [--TRAIN_SIZE=<TRAIN_SIZE>] [--NEW_GEN=<NEW_GEN>] [--NEW_GEN_PATH=<NEW_GEN_PATH>]

Options:
--DATA_FILE_PATH=<DATA_FILE_PATH>       Path (including filename) to retrieve the csv file. [default: data/pokemon_smogon_competitive.csv]
--TRAIN_FILE_PATH=<TRAIN_FILE_PATH>     Path (including filename) to print the train portion as a csv file. [default: data/pokemon_smogon_competitive_train.csv]
--TEST_FILE_PATH=<TEST_FILE_PATH>       Path (including filename) to print the test portion as a csv file. [default: data/pokemon_smogon_competitive_test.csv]
--TRAIN_SIZE=<TRAIN_SIZE>               Decimal value for the train/test split. [default: 0.75]
--NEW_GEN=<NEW_GEN>                     TRUE if the dataset contains info about the new generation (final test), FALSE otherwise [default: False]
--NEW_GEN_PATH=<NEW_GEN_PATH>           Path (including filename) to print the wrangled new gen dataset as a csv file. This only applies if NEW_GEN is True [default: data/new_gen_wrangled.csv]
'''

import os
import pandas as pd
from sklearn.model_selection import train_test_split 
from docopt import docopt

opt = docopt(__doc__)


def main(data_file_path, train_file_path, test_file_path, train_size, new_gen, new_gen_path):
    """
    Main entry for the data download script.

    Arguments
    ---------
    data_file_path : str
        File path (including filename) to retrieve the data file from.
    train_file_path : str
        File path (including filename) to print the train portion of the data.
    test_file_path : str
        File path (including filename) to print the test portion of the data.
    train_size : float
        Size of the train dataset.
    """
    print("Checking the path of the data... \n", end='')
    loaded_df = load_data(data_file_path)
    print("Wrangling the data... \n", end='')    
    wrangled_df = wrangling(loaded_df)
    
    # This portion is for the original dataset
    if (new_gen == 'False'):
        print("Splitting the data... \n", end='')
        train, test = train_test_split(wrangled_df, train_size = float(train_size), test_size = 1 - float(train_size), random_state = 2020)

        print("Saving the train data... \n", end='')
        train.to_csv(train_file_path)
        print("Saving the test data... \n", end='')
        test.to_csv( test_file_path)

    else:
        print("Saving the new gen data... \n", end='')
        wrangled_df.to_csv(new_gen_path)

        

def load_data(data_file_path):
    """
    Loads the data given a data file path.

    Arguments
    ---------
    data_file_path : str
        File path (including filename) to retrieve the data file from.

    Returns
    ---------
    data : pandas dataframe
        Loaded data as a pandas dataframe.
    """
    assert os.path.isfile(data_file_path), "File does not exist"
    
    return(pd.read_csv(data_file_path))

def wrangling(data):
    """
    Wrangles and splits the data.

    Arguments
    ---------
    data : Pandas Dataframe
        Pandas dataframe for the data wrangling and split.

    Returns
    ---------
    data : pandas dataframe
        Wrangled dataset.
    """
    # Ubers
    data['Tier_2'] = data['Tier'].str.replace("^AG$", "AG - Ubers")
    data['Tier_2'] = data['Tier_2'].str.replace("^Uber$", "AG - Ubers")

    # Upper tier
    data['Tier_2'] = data['Tier_2'].str.replace("^OU$", "Upper Tiers")
    data['Tier_2'] = data['Tier_2'].str.replace("^BL$", "Upper Tiers")
    data['Tier_2'] = data['Tier_2'].str.replace("^UU$", "Upper Tiers")

    # Lower tier
    data['Tier_2'] = data['Tier_2'].str.replace("^BL2$", "Lower Tiers")
    data['Tier_2'] = data['Tier_2'].str.replace("^RU$", "Lower Tiers")
    data['Tier_2'] = data['Tier_2'].str.replace("^BL3$", "Lower Tiers")
    data['Tier_2'] = data['Tier_2'].str.replace("^NU$", "Lower Tiers")
    data['Tier_2'] = data['Tier_2'].str.replace("^BL4$", "Lower Tiers")
    data['Tier_2'] = data['Tier_2'].str.replace("^PU$", "Lower Tiers")

    # Has Sencodary Type?
    data['Has_ST'] = data['Type.2'].apply(lambda x: 'No' if pd.isnull(x) else 'Yes')
    
    data = data.rename(columns = {'X.': 'Number',
                                  'Type.1': 'Type1',
                                  'Type.2': 'Type2',
                                  'Sp..Atk': 'Special_attack',
                                  'Sp..Def': 'Special_defense'})

    return(data)

if __name__ == "__main__":
    main(opt["--DATA_FILE_PATH"],
         opt["--TRAIN_FILE_PATH"],
         opt["--TEST_FILE_PATH"],
         opt["--TRAIN_SIZE"],
         opt["--NEW_GEN"],
         opt["--NEW_GEN_PATH"])
