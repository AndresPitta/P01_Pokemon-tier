# authors: Andres Pitta
# date: 2020-02-23

'''This script will generate exploratory data analysis visualizations. It takes as arguments the file were the root 
file is, the path where the visualizations will be saved.

Usage: eda.py [--DATA_FILE_PATH=<DATA_FILE_PATH>] [--EDA_FILE_PATH=<EDA_FILE_PATH>]

Options:
--DATA_FILE_PATH=<DATA_FILE_PATH>  Path (including filename) to gather the csv file. [default: data/pokemon_smogon_competitive_train.csv]
--EDA_FILE_PATH=<EDA_FILE_PATH>  Path to output EDA files. [default: results/figures/]
'''

from docopt import docopt
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from selenium import webdriver
import os
#browser = webdriver.Chrome('C:\webdrivers\chromedriver.exe')


opt = docopt(__doc__)

def main(data_file_path, eda_file_path):
    assert os.path.isfile(data_file_path), "File does not exist"
    assert os.path.isdir(eda_file_path), "EDA_FILE_PATH does not exist, please create a 'figures' folder in results"

    data = pd.read_csv(data_file_path)
    
    make_correlation(data, eda_file_path)
    make_bars(data, eda_file_path)

def make_correlation(data, eda_file_path):
    """
    Creates a pearson's correlation plot of the continuous variables.

    Parameters:
    data -- (dataframe) The training data
    eda_file_path -- (str) The path to specify where the plot is saved
    """

    data_corr = (data
                 .corr()
                 .reset_index()
                 .rename(columns = {'index':'Variable 1'})
                 .melt(id_vars = ['Variable 1'],
                       value_name = 'Correlation',
                       var_name = 'Variable 2')
                )

    base = alt.Chart(data_corr).encode(
        alt.Y('Variable 1:N'),
        alt.X('Variable 2:N')
    ) 

    heatmap = base.mark_rect().encode(
        alt.Color('Correlation:Q',
                    scale=alt.Scale(scheme='viridis'))
    )

    text = base.mark_text(baseline='middle').encode(
        text=alt.Text('Correlation:Q', format='.2'),
        color=alt.condition(
            alt.datum.Correlation >= 0.95,
            alt.value('black'),
            alt.value('white')
        )
    )

    plot = (heatmap + text).properties(
        width = 400,
        height = 400,
        title = "Pearson's Correlation"
    ).configure_axis(labelFontSize=15, 
                                    titleFontSize=22
                                    ).configure_title(fontSize=26)

    plot.save("{}corrplot.png".format(eda_file_path))
    print(f"corrplot.png saved to {eda_file_path}")


def make_bars(data, eda_file_path):
    """
    Creates bar plots of average stats vs. the response variable.

    Parameters:
    data -- (dataframe) The training data
    eda_file_path -- (str) The path to specify where the plot is saved
    """

    numerical_features = ['HP', 'Attack', 'Defense', 'Special_attack', 'Special_defense', 'Speed']
    numerical_encodings = ['HP', 'Attack', 'Defense', 'Special_attack', 'Special_defense', 'Speed']

    for i in range(len(numerical_features)):
        pokemon_graph = data[['Tier_2', numerical_features[i]]].groupby(by = 'Tier_2')\
                                                                        .mean()\
                                                                        .reset_index()
        

        chart = alt.Chart(pokemon_graph).mark_bar().encode(
            alt.X(numerical_features[i]),
            alt.Y('Tier_2:O', 
                  sort = alt.EncodingSortField(field = numerical_features[i], order = "descending"),
                  axis = alt.Axis(title = "Tier classification", tickCount = 8))
        ).properties(width=500, 
                    height=500, 
                    title = f'Mean {numerical_features[i]} by Tier'
                    ).configure_axis(labelFontSize=15, 
                                    titleFontSize=22
                                    ).configure_title(fontSize=26)
        
        chart.save('{}{}.png'.format(eda_file_path, numerical_encodings[i]))
        print(f"{numerical_encodings[i]}.png saved to {eda_file_path}")

if __name__ == "__main__":
     main(opt["--DATA_FILE_PATH"], opt["--EDA_FILE_PATH"])