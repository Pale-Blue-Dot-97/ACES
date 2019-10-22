"""starting_point.py

Script to load Voyager magnetometer data in from file and clean, interpolate and work on

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

# Load in data from PDS archive. Column headers come from the LBL meta data
data = pd.read_table('S3_1_92S.TAB', delimiter=',',
                     names=['TIME', 'SCLK', 'MAG_ID', 'BR', 'BTH', 'BPH', 'BMAG',
                            'AVG_BMAG', 'DELTA', 'LAMBDA', 'RMS_BR', 'RMS_BTH', 'RMS_BPH', 'NUM_PTS'])

# Find missing data points
# pp_real = data['BR']>-999.0
# pp_missing = data['BR']<-999.0

# Interpolate missing data
# ii = np.arange(1,len(data),1)
# fun = scipy.interpolate.interp1d(ii[pp_real], data['BR'][pp_real])

# datetime.datetime.strptime(time_string, '%Y-%m-DT%h:%....')

plt.subplot(4, 1, 1)
plt.plot(data['BR'])
plt.ylabel('B_r [nT]')
plt.subplot(4, 1, 2)
plt.plot(data['BTH'])
plt.ylabel('B_th [nT]')
plt.subplot(4, 1, 3)
plt.plot(data['BPH'])
plt.ylabel('B_ph [nT]')
plt.subplot(4, 1, 4)
plt.plot(data['BMAG'])
plt.ylabel('|B| [nT]')
plt.show()
