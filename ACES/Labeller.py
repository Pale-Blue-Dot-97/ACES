"""Script to manually label data for neural network classification

TODO:
    * Generalise for all ACES magnetometer data

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import sys

# =====================================================================================================================
#                                                     GLOBAL
# =====================================================================================================================
pd.plotting.register_matplotlib_converters()

# Header names in the labels file
header = ['CLASS', 'START', 'STOP']


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_data(filename):
    """Load in cleaned and normalised data from file

    Args:
        filename (str): Name of file to be loaded

    Returns:
        data (DataFrame): Table of all the cleaned and normalised data from file
        norm_data (DataFrame): Table of just the normalised data

    """

    data_names = ['BR', 'BTH', 'BPH', 'BMAG', 'UNIX TIME', 'BR_norm', 'BTH_norm', 'BPH_norm', 'BMAG_norm']

    data = pd.read_csv(filename, names=data_names, dtype=float, header=0)\
        .drop(columns=['BR', 'BTH', 'BPH', 'BMAG'])

    # Create Matplotlib datetime64 type date-time column from UNIX time
    data['DATETIME'] = pd.to_datetime(data['UNIX TIME'], unit='s')

    # Re-index data to date-time
    data.index = data['DATETIME']
    del data['DATETIME']

    return data


def load_labels(data_filename, labels_filename):
    """

    Args:
        data_filename (str): Name of file containing data
        labels_filename (str): Name of file containing labels

    Returns:
        labelled_data (DataFrame): Labelled data
        classes ([str]): List of class names found in labels file

    """

    # Loads in data
    data = load_data(data_filename)

    # Loads the start and endpoints of the labelled regions of the data
    labels = pd.read_csv(labels_filename, names=header, dtype=str, header=0, sep=',', index_col='CLASS')

    # Converts strings to datetime64 dtype
    labels['START'] = pd.to_datetime(labels['START'])
    labels['STOP'] = pd.to_datetime(labels['STOP'])

    # Creates new DataFrame with an additional column for the class labels
    labelled_data = data.copy()
    labelled_data['LABELS'] = float('NaN')

    # Dict to hold all the individual class DataFrame slices
    LABELS = {}

    # Finds class names from Labels
    classes = [item[0] for item in Counter(labels.index).most_common()]

    # Slices labels into separate DataFrames for each classification
    for classification in classes:
        LABELS[classification] = labels.loc[classification].reset_index(drop=True)

    for classification in classes:
        label_match_list = []

        class_df = LABELS[classification]

        for i in range(len(class_df)):
            # Start and endpoints for a label range
            start = class_df['START'][i]
            stop = class_df['STOP'][i]

            # Finds all data points that lie in that datetime range and adds to list
            label_match_list = label_match_list + labelled_data[start:stop].index.tolist()

        # Classifies all data points that lie in this classified event ranges
        labelled_data['LABELS'][label_match_list] = classification

    # Labels any remaining points as False
    labelled_data['LABELS'].fillna(False, inplace=True)

    return labelled_data, classes


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    data, classes = load_labels(sys.argv[1], sys.argv[2])

    # Plot using inbuilt Pandas function
    data.plot(y=['BR_norm', 'BMAG_norm'], kind='line')

    for classification in classes:
        plt.plot(data.loc[data['LABELS'] == classification].index.to_list(),
                 (data.loc[data['LABELS'] == classification]['BMAG_norm']), 'o', ms=0.5, alpha=0.8)

    plt.legend(['BR', 'BMAG'] + classes)
    plt.show()


if __name__ == '__main__':
    main()
