"""Script to process cleaned and normalised Voyager magnetometer data into 4-channel blocks for use as training and
validation data for neural networks

TODO:
    * Add methods to process data into 4-channel blocks for training
    * Add method to automatically label data

"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import time
import sys
import pandas as pd
import numpy as np
import scipy.interpolate as ip
import scipy.signal as sg
import datetime
import Plot2D as laplt
import MultiFig as mf
import pyttsx3 as speech

# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================


def load_data():
    """Load in cleaned and normalised data from file
    """

    data_names = ['BR', 'BTH', 'BPH', 'BMAG', 'UNIX TIME', 'BR_norm', 'BTH_norm', 'BPH_norm']

    data = pd.read_csv('VOY2_JE_PROC.csv', names=data_names)

    return data


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    engine = speech.init()

    engine.say("Loading data")
    engine.runAndWait()

    data = load_data()

    # Alert bell
    for i in range(1, 4):
        sys.stdout.write('\r\a')
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write('\n')

    engine.say("Finished")
    engine.runAndWait()


if __name__ == '__main__':
    main()
