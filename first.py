import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

layer = Dense(units = 1, input_shape = [1])
model = Sequential([layer])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1, 0, 1,2,3,4])
ys = np.array([-3,-1,1,3,5,7])
model.fit(xs, ys, epochs=500)
print(model.predict([10]))
print(layer.get_weights())