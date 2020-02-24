"""Script to manually label Voyager 1 Jupiter flyby magnetometer data for neural network training

TODO:
    * Import Voyager 1 data and plot against data index and datetime
    * Manually define CSCs and output to file for data labelling
    * Add further science classifications

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import time
from datetime import datetime
import sys
import pandas as pd
import numpy as np
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import scipy.signal as sg
import random


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

    data = pd.read_csv('VOY2_JE_PROC.csv', names=data_names, dtype=float, header=0)

    norm_data = data.drop(columns=['BR', 'BTH', 'BPH', 'BMAG'])

    norm_data.reset_index(drop=True)
    data.reset_index(drop=True)

    return data, norm_data


def unix_to_datetime(times):
    times.to_list()

    datetimes = []

    for i in times:
        datetimes.append(datetime.utcfromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S'))

    print(datetimes[0])

    #plotable = dates.date2num(datetimes[0])
    #print(plotable)

    return datetimes


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    # Load all data and normalised data from file in Pandas.DataFrame form
    data, norm_data = load_data()

    norm_data['DATETIME'] = unix_to_datetime(norm_data['UNIX TIME'])

    p = int(len(norm_data['BR_norm'])/10)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    # Plot B_r against date-time
    ax1.plot_date(norm_data['DATETIME'], norm_data['BR_norm'], fmt='-')
    ax1.set_xticks(np.arange(0, len(norm_data['BR_norm']), p))
    ax1.set_xlabel('Date-time (UTC)')
    ax1.set_ylabel('B_r')

    # Add second x-axis for index
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(range(0, len(norm_data['BR_norm']), p))
    ax2.set_xticklabels(range(0, len(norm_data['BR_norm']), p))
    ax2.set_xlabel('Index Location')

    plt.show()


if __name__ == '__main__':
    pd.plotting.register_matplotlib_converters()
    main()
