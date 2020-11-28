import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import time
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

(x_train, y_train), (x_test, y_test) = mnist.load_data()

unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))

num_labels = len(np.unique(y_train))
start_time = time.time()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, activation="relu", input_shape=(image_size, image_size, 1)))
model.add(MaxPooling2D(2))
model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
model.add(MaxPooling2D(2))
model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(num_labels))
model.add(Activation("softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=1, batch_size=128)

loss, acc = model.evaluate(x_test, y_test, batch_size=128)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))
print(f"Elapsed time: {time.time()-start_time} sec.")

predictions = model.predict(x_test)

# sample 25 mnist digits from train dataset
indexes = np.random.randint(0, x_test.shape[0], size=25)
images = x_test[indexes]
labels = y_test[indexes]

# plot the 25 mnist digits
plt.figure(figsize=(15, 15))
plt.rc("font", size=14)
for i in range(len(indexes)):
    plt.subplot(5, 5, i + 1)
    image = images[i]
    plt.imshow(image, cmap="gray")
    plt.title(f"Label:{np.argmax(labels[i])}, Predict:{np.argmax(predictions[indexes[i]])}")
    plt.axis("off")

plt.savefig("cnn-mnist.png")
plt.show()
plt.close("all")
