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
import pyttsx3 as speech

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

    data = pd.read_csv('VOY2_JE_PROC.csv', names=data_names)

    norm_data = data.drop(columns=['BR', 'BTH', 'BPH', 'BMAG'])

    return data, norm_data


# =====================================================================================================================
#                                                       MAIN
# =====================================================================================================================
def main():
    engine = speech.init()

    engine.say("Loading data")
    engine.runAndWait()

    data, norm_data = load_data()

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
