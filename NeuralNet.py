from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time

#Data Import

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x= (x/255).astype("float32")
y = to_categorical(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)
