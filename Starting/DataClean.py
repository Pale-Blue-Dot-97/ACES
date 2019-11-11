"""DataClean.py

Script to load Voyager magnetometer data in from file and clean, interpolate and work on

TODO:
    * Convert to using DataLoad to load data to pandas DataFrame
    * Compute variances between points to clean non-physical data
    * Use variances to calculate transmission priorities/ compression or fitting levels
"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal as sg
import datetime
import Plot2D as laplt
import MultiFig as mf

# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================


def load_data():
    """Load in data from PDS archive. Column headers come from the LBL meta data
    """
    names = ['TIME', 'SCLK', 'MAG_ID', 'BR', 'BTH', 'BPH', 'BMAG', 'AVG_BMAG', 'DELTA', 'LAMBDA', 'RMS_BR', 'RMS_BTH',
             'RMS_BPH', 'NUM_PTS']

    data = pd.read_csv('S3_1_92S.TAB', names=names, na_values=-9999.999)

    data_columns = ['BR', 'BTH', 'BPH', 'BMAG', 'AVG_BMAG', 'DELTA', 'LAMBDA', 'RMS_BR', 'RMS_BTH', 'RMS_BPH',
                    'NUM_PTS']

    return data, data_columns


def calc_variances(data, data_columns):
    deleted = []
    for i in data_columns:
        data_min = np.min(data[i])
        data_max = np.max(data[i])
        max_var = 0.5 * np.abs(data_max - data_min)
        for j in range(len(data[i])):
            if j != 0 and j not in deleted:
                this_point = data[i][j]
                k = j-1
                while k in deleted:
                    k = k-1

                last_point = data[i][k]
                var = np.abs(this_point - last_point)
                if var > max_var:
                    print('Deleted entry %d' % j)
                    print('Last point: %d' % k)
                    print(this_point)
                    print(last_point)
                    deleted.append(j)
                    data.drop(j, axis=0, inplace=True)


def medfilt_data(data, data_columns, kernel_size):
    data = data.copy()

    cleaned_data_arrays = []

    for i in data_columns:
        print('Filtering %s' % i)
        filtered = []
        for j in np.arange(kernel_size, kernel_size + 20, 2):
            print('Kernel Size: %s' % j)
            for k in [1, 2, 3]:
                filtered = sg.medfilt(data[i], j)

        cleaned_data_arrays.append(filtered)
    new_data = {}

    for i in range(len(data_columns)):
        new_data.update({data_columns[i]: cleaned_data_arrays[i]})

    cleaned_df = pd.DataFrame(new_data)

    return cleaned_df


def clean_data(data, data_columns, kernel_size):
    data = data.copy()
    deleted = []
    for i in data_columns:
        data_min = np.min(data[i])
        data_max = np.max(data[i])
        max_var = 0.1 * np.abs(data_max - data_min)

        noise = sg.find_peaks(data[i], threshold=max_var, width=(1, 5), wlen=kernel_size)

        print(noise[0])

        for j in noise[0]:
            if j not in deleted:
                print('Deleted %d' % j)
                deleted.append(j)
                data.drop(data.index[j], axis=0, inplace=True)

    return data, deleted


def main():
    data, data_columns = load_data()

    time = []

    print('Extracting time from time-stamps')

    skipped = []

    for i in range(len(data['TIME'])):
        j = data['TIME'][i]
        print(j)

        if j[17] == '6':
            s = list(j)
            s[17] = '0'
            s[14:15] = '%s' % str(float(j[14:15])+1)

            if s[14] == '6':
                s[14] = '0'
                s[11:12] = '%s' % str(float(j[11:12]) + 1)

                if s[11] == '2' and s[12] == '4':
                    s[11] = '0'
                    s[12] = '0'
                    s[8:9] = '%s' % str(float(j[8:9]) + 1)

        try:
            t = datetime.datetime.strptime(j, '%Y-%m-%dT%H:%M:%S.%f')
            time.append(t)

        except ValueError:
            print('Exception in datestamp: Removing row %d' % i)
            skipped.append(i)
            #data.drop(data.index[i], axis=0, inplace=True)

    print(skipped)

    raw_data = data.copy()
    med_data = medfilt_data(data, data_columns, 5)

    t = np.linspace(start=0, stop=len(raw_data['BR']), num=len(raw_data['BR']))

    laplt.create_figure(y=[med_data['BR'], raw_data['BR']], x=[t, t], figure_name='raw_vs_filtered.png',
                        COLOURS=['r', 'k'], POINTSTYLES=['-'], DATALABELS=['Filtered Data', 'Raw Data'], x_label='Time',
                        y_label='B_r [nT]')


if __name__ == '__main__':
    main()
