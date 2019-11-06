"""starting_point.py

Script to load Voyager magnetometer data in from file and clean, interpolate and work on

TODO:
    * Compute variances between points to clean non-physical data
    * Switch to using LancAstro.py's MultFig.py for plotting (may as well make use of it)
    * Use variances to calculate
"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal as sg

# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================


def load_data():
    """Load in data from PDS archive. Column headers come from the LBL meta data
    """
    names = ['TIME', 'SCLK', 'MAG_ID', 'BR', 'BTH', 'BPH', 'BMAG', 'AVG_BMAG', 'DELTA', 'LAMBDA', 'RMS_BR', 'RMS_BTH',
         'RMS_BPH', 'NUM_PTS']

    data = pd.read_table('S3_1_92S.TAB', delimiter=',', names=names, na_values=-9999.999)

    data_columns = ['BR', 'BTH', 'BPH', 'BMAG', 'AVG_BMAG', 'DELTA', 'LAMBDA', 'RMS_BR', 'RMS_BTH', 'RMS_BPH', 'NUM_PTS']

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

# def smooth_data()


def medfilt_data(data, data_columns, kernel_size):
    data = data.copy()

    cleaned_data_arrays = []

    for i in data_columns:
        print('Filtering %s' % i)
        filtered = []
        for j in np.arange(kernel_size, kernel_size + 20, 2):
            print('Kernel Size: %s' % j)
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


# datetime.datetime.strptime(time_string, '%Y-%m-DT%h:%....')


def main():
    #calc_variances(data)
    data, data_columns = load_data()

    raw_data = data.copy()
    #cleaned_data = clean_data(data, data_columns, 21)
    med_data = medfilt_data(data, data_columns, 5)

    #print("Length of Raw BR: %s" % len(raw_data['BR']))
    #print("Length of Clean BR: %s" % len(cleaned_data['BR']))

    plt.subplot(5, 2, 1)
    plt.plot(med_data['BR'])
    plt.ylabel('B_r [nT]')

    plt.subplot(5, 2, 2)
    plt.plot(raw_data['BR'])
    plt.ylabel('RAW_B_r [nT]')

    plt.show()


if __name__ == '__main__':
    main()
