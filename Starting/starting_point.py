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

# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================

# Load in data from PDS archive. Column headers come from the LBL meta data
names = ['TIME', 'SCLK', 'MAG_ID', 'BR', 'BTH', 'BPH', 'BMAG', 'AVG_BMAG', 'DELTA', 'LAMBDA', 'RMS_BR', 'RMS_BTH',
         'RMS_BPH', 'NUM_PTS']

data = pd.read_table('S3_1_92S.TAB', delimiter=',', names=names, na_values=-9999.999)

data_columns = ['BR', 'BTH', 'BPH', 'BMAG', 'AVG_BMAG', 'DELTA', 'LAMBDA', 'RMS_BR', 'RMS_BTH', 'RMS_BPH', 'NUM_PTS']


def calc_variances(data):
    deleted = []
    for i in data_columns:
        data_min = np.min(data[i])
        data_max = np.max(data[i])
        max_var = 0.5 * np.abs(data_max - data_min)
        for j in range(len(data[i])):
            if j != 0:
                this_point = data[i][j]
                k = j-1
                while k in deleted:
                    k = k-1

                last_point = data[i][k]
                var = np.abs(this_point - last_point)
                if var > max_var:
                    print('Delete %d' % j)
                    print(this_point)
                    print(last_point)
                    deleted.append(j)
                    data.drop(j, axis=0, inplace=True)

# Interpolate missing data
# ii = np.arange(1,len(data),1)
# fun = scipy.interpolate.interp1d(ii[pp_real], data['BR'][pp_real])

# datetime.datetime.strptime(time_string, '%Y-%m-DT%h:%....')


def main():
    calc_variances(data)

    plt.subplot(4, 2, 1)
    plt.plot(data['BR'])
    plt.ylabel('B_r [nT]')
    plt.subplot(4, 2, 2)
    plt.plot(data['BTH'])
    plt.ylabel('B_th [nT]')
    plt.subplot(4, 2, 3)
    plt.plot(data['BPH'])
    plt.ylabel('B_ph [nT]')
    plt.subplot(4, 2, 4)
    plt.plot(data['BMAG'])
    plt.ylabel('|B| [nT]')
    plt.show()


if __name__ == '__main__':
    main()
