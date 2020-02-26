"""Script to manually label Voyager 1 Jupiter flyby magnetometer data for neural network training

TODO:
    * Label all of data using class start/stop points

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================================================================
#                                                     GLOBAL
# =====================================================================================================================
pd.plotting.register_matplotlib_converters()


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_data():
    """Load in cleaned and normalised data from file

    Returns:
        data (DataFrame): Table of all the cleaned and normalised data from file
        norm_data (DataFrame): Table of just the normalised data

    """

    data_names = ['BR', 'BTH', 'BPH', 'BMAG', 'UNIX TIME', 'BR_norm', 'BTH_norm', 'BPH_norm', 'BMAG_norm']

    data = pd.read_csv('VOY2_JE_PROC.csv', names=data_names, dtype=float, header=0)\
        .drop(columns=['BR', 'BTH', 'BPH', 'BMAG'])

    # Create Matplotlib datetime64 type date-time column from UNIX time
    data['DATETIME'] = pd.to_datetime(data['UNIX TIME'], unit='s')

    # Re-index data to date-time
    data.index = data['DATETIME']
    del data['DATETIME']

    return data


def load_labels(data):
    # List of class names
    classes = ['CSC', 'NSC', 'MP']

    header = ['CLASS', 'START', 'STOP']

    # Loads the start and endpoints of the labelled regions of the data
    labels = pd.read_csv('Labels.csv', names=header, dtype=str, header=0, sep=',', index_col='CLASS')

    print(labels)

    # Converts strings to datetime64 dtype
    labels['START'] = pd.to_datetime(labels['START'])
    labels['STOP'] = pd.to_datetime(labels['STOP'])

    # Creates new DataFrame with an additional column for the class labels
    labelled_data = data.copy()
    labelled_data['LABELS'] = float('NaN')

    # Dict to hold all the individual class DataFrame slices
    LABELS = {}

    # Slices labels into separate DataFrames for each classification
    for classification in classes:
        LABELS[classification] = labels.loc[classification].reset_index(drop=True)

    for classification in classes:
        label_match_list = []

        class_df = LABELS[classification]

        for i in range(len(class_df)):
            start = class_df['START'][i]
            stop = class_df['STOP'][i]

            label_slice = labelled_data[start:stop]
            label_slice_index = label_slice.index.tolist()

            label_match_list = label_match_list + label_slice_index

    return labelled_data


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    # Load all data and normalised data from file in Pandas.DataFrame form
    data = load_data()

    load_labels(data)

    # Plot using inbuilt Pandas function
    data.plot(y=['BR_norm', 'BMAG_norm'], kind='line')

    plt.show()


if __name__ == '__main__':
    main()
