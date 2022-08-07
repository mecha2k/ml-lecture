import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from icecream import ic

model = Sequential()
model.add(Dense(24, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.build(input_shape=(None, 3))
ic(len(model.weights))

model = Sequential()
model.add(Input(shape=(3,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.summary()

ic(tf.shape(1.0))
con = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
ic(tf.shape(con))

a = Input(shape=(None, 10))
ic(tf.shape(a))
ic(a.shape)

a = Input(shape=(28, 28))
ic(tf.shape(a))
ic(a.shape)
