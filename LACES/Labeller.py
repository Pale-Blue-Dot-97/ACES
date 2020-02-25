"""Script to manually label Voyager 1 Jupiter flyby magnetometer data for neural network training

TODO:
    * Manually define CSCs and output to file for data labelling
    * Add further science classifications

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


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    # Load all data and normalised data from file in Pandas.DataFrame form
    data = load_data()

    # Plot using inbuilt Pandas function
    ax = data.plot(y='BR_norm', kind='line')

    # Create secondary x-axis for index location ticks
    ax2 = ax.twiny()
    ax2.set_xticks(range(0, len(data['BR_norm']), 100000))

    plt.show()


if __name__ == '__main__':
    main()
