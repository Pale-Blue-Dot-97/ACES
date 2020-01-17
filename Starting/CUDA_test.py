import tensorflow as tf
#from tensorflow.keras import layers, models

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
