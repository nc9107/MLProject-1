import pandas as pd
import numpy as np
from IPython.core.pylabtools import figsize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# Function to calculate missing values by column
def missing_values_tables(df):
    # Total missing values
    missing_val_num = df.isnull().sum()

    # Percentage of missing values
    missing_val_percent = 100 * missing_val_num / len(df)

    # Making a table with those results
    missing_val_table = pd.concat([missing_val_num, missing_val_percent], axis=1)

    # Renaming the columns
    missing_val_table_rename_cols = missing_val_table.rename(columns={0: 'Missing values', 1: '% of Total Values'})

    # Sorting the table by percentage of missing. Descending order.
    missing_val_table_rename_cols = missing_val_table_rename_cols[
        missing_val_table_rename_cols.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n" "There are " + str(
        missing_val_table_rename_cols.shape[0]) +
          " columns that have missing values.")

    # Returning the dataframe with missing information
    return missing_val_table_rename_cols


def main():
    pd.set_option('display.max_columns', 60)
    # Read in data into data frames
    # data = pd.read_csv('data/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv')
    # data = pd.read_csv('C:\\Users\\Venkata\\Desktop\\MLProject-data-1.csv')
    data = pd.read_csv('MLProject-data-1.csv')

    # Display top of dataframe
    # data.head()
    #
    # data.info()

    # Replace all occurrences of Not Available with numpy not a number
    data = data.replace({'Not Available': np.nan})

    # Iterate through the columns
    for col in list(data.columns):
        # Select columns that should be numeric
        if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in
                col or 'therms' in col or 'gal' in col or 'Score' in col):
            # Convert the data type to float
            data[col] = data[col].astype(float)

    # print(data.describe())
    # missing_values_tables(data)

    # Getting columns with more than 50% of missing data
    missing_df = missing_values_tables(data);
    # print("**********************")
    # print(missing_df)
    missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)
    # print(" We will remove {} columns. ".format( len(missing_columns)))

    # Dropping the columns
    data = data.drop(columns=list(missing_columns))

    ############################################################################################
    # ENTERING DATA ANALYSIS
    """
    Exploratory Data Analysis (EDA) is an open-ended process where we make plots and calculate statistics
    in order to explore our data. The purpose is to to find anomalies, patterns, trends, or relationships. 
    These may be interesting by themselves (for example finding a correlation between two variables) or 
    they can be used to inform modeling decisions such as which features to use. In short, the goal of EDA 
    is to determine what our data can tell us! EDA generally starts out with a high-level overview, and then narrows 
    in to specific parts of the dataset once as we find interesting areas to examine.

    """
    # An univariate plot shows the distribution shows the distribution of a single variable such as in histogram.

    figsize(8, 8)

    # print(data)

    # Rename the score
    data = data.rename(columns={'ENERGY STAR Score': 'score'})
    # print("//////////////////")
    # print(data)

    # Histogram of the Energy Star Score
    plt.style.use('fivethirtyeight')
    plt.hist(data['score'].dropna(), bins=100, edgecolor='k');
    plt.xlabel('Score');
    plt.ylabel('Number of Buildings');
    plt.title('Energy Star Score Distribution');
    # plt.show()

    """
    Energy Use Intensity (EUI), which is the total energy use divided
     by the square footage of the building. Here the energy usage is 
     not self-reported, so this could be a more objective measure of
     the energy efficiency of a building. Moreover, this is not a percentile 
     rank, so the absolute values are important and we would expect them to be 
     approximately normally distributed with perhaps a few outliers on the low 
     or high end
    """

    # Histogram plot site of EUI
    figsize(8, 8)
    plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins=20, edgecolor='black');
    plt.xlabel('Site EUI');
    plt.ylabel('Count');
    plt.title('Site EUI distribution')
    # plt.show()

    # print("/////////////////////////*****************************")
    # print(data['Site EUI (kBtu/ft²)'].describe())

    # Getting the buildings with the highest energy ratings.
    #  print(data['Site EUI (kBtu/ft²)'].dropna().sort_values().tail(10))

    # Extracting that column from the data set.
    # print(data.loc[data['Site EUI (kBtu/ft²)'] == 869265.0,:])
    """
    Removing Outliers
    When we remove outliers, we want to be careful that we are not throwing 
    away measurements just because they look strange. They may be the result 
    of actual phenomenon that we should further investigate. When removing 
    outliers, I try to be as conservative as possible, using the definition of
     an extreme outlier:

    On the low end, an extreme outlier is below:
     $\text{First Quartile} -3 * \text{Interquartile Range}$

    On the high end, an extreme outlier is above:
     $\text{Third Quartile} + 3 * \text{Interquartile Range}$
    """
    # Calculating first and third quartile
    first_quartile = data['Site EUI (kBtu/ft²)'].describe()['25%']
    third_quartile = data['Site EUI (kBtu/ft²)'].describe()['75%']

    # Calculating interquartile range
    iqr = third_quartile - first_quartile

    # Removing outliers data = data
    data = data[(data['Site EUI (kBtu/ft²)'] > (first_quartile - 3 * iqr)) &
                (data['Site EUI (kBtu/ft²)'] < (third_quartile + 3 * iqr))]

    # Plotting a revised SITE EUI Distribution s
    figsize(8, 8)
    plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins=20, edgecolor='black');
    plt.xlabel('Site EUI');
    plt.ylabel('Count');
    plt.title('Site EUI Distribution');
    plt.show()

# db.Dogs.find({"fleas":{$gt:5}})
if __name__ == '__main__':
    main()
