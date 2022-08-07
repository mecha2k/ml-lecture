import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.data import Dataset
from tensorflow.keras.datasets import mnist


np.random.seed(42)
plt.style.use("seaborn-white")

epochs = 1000
learning_rate = 0.1

X = np.random.randn(50)
y = 2 * X + np.random.randn(50)

W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

optimizer = tf.optimizers.SGD(learning_rate)


def linear_regression(X):
    return X * W + b


def compute_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def run_optimization():
    with tf.GradientTape() as tape:
        y_pred = linear_regression(X)
        loss = compute_loss(y, y_pred)

    gradients = tape.gradient(target=loss, sources=[W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


# for epoch in range(epochs):
#     run_optimization()
#     if epoch % 100 == 0:
#         pred = linear_regression(X)
#         loss = compute_loss(y, pred)
#         print(f"Epoch: {epoch}, loss: {loss:.4f}, W: {W.numpy():.4f}, b: {b.numpy():.4f}")


a = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())
c = tf.Variable(np.random.randn())

y = X**2 + X * np.random.randn(50)

line_x = np.arange(min(X), max(X), 0.1)
line_y = a * line_x**2 + b * line_x + c

X_ = np.arange(-4, 4, 0.1)
y_ = a * X_**2 + b * X_ + c

optimizer = Adam(learning_rate=learning_rate)

# for epoch in range(epochs):
#     with tf.GradientTape() as tape:
#         y_pred = a * X**2 + b * X + c
#         loss = compute_loss(y, y_pred)

#     gradients = tape.gradient(target=loss, sources=[a, b, c])
#     optimizer.apply_gradients(zip(gradients, [a, b, c]))

#     if epoch % 100 == 0:
#         pred = a * X**2 + b * X + c
#         loss = compute_loss(y, pred)
#         print(
#             f"Epoch: {epoch}, loss: {loss:.4f}, a: {a.numpy():.4f}, b: {b.numpy():.4f}, c: {c.numpy():.4f}"
#         )


def compute_loss1():
    y_pred = a * X**2 + b * X + c
    return tf.reduce_mean(tf.square(y - y_pred))


# for epoch in range(epoch):
#     optimizer.minimize(compute_loss1, var_list=[a, b, c])
#     if epoch % 100 == 0:
#         pred = a * X**2 + b * X + c
#         loss = compute_loss(y, pred)
#         print(
#             f"Epoch: {epoch}, loss: {loss:.4f}, a: {a.numpy():.4f}, b: {b.numpy():.4f}, c: {c.numpy():.4f}"
#         )

# y_ = a * X_**2 + b * X_ + c

# plt.scatter(X, y, label="data")
# plt.plot(line_x, line_y, "g--", label="model")
# plt.plot(X_, y_, "r--", label="prediction")
# plt.xlim(-4, 4)
# plt.legend()
# plt.grid()
# plt.show()


(x_train, y_train), (x_test, y_test) = mnist.load_data()

input_shape = x_train.shape[1:]
num_features = input_shape[0] * input_shape[1]
num_classes = np.unique(y_train).shape[0]
print("input shape:", input_shape)
print("num_classes:", num_classes)
print("num_features:", num_features)

x_train = np.array(x_train[:20000], dtype=np.float32).reshape(-1, num_features) / 255.0
x_test = np.array(x_test[:20000], dtype=np.float32).reshape(-1, num_features) / 255.0
y_train = np.array(y_train[:20000], dtype=np.int32)
y_test = np.array(y_test[:20000], dtype=np.int32)

y_train_onehot = to_categorical(y_train, num_classes)
y_test_onehot = to_categorical(y_test, num_classes)
print("y_train_onehot shape:", y_train_onehot.shape)

train_dataset = Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.repeat().shuffle(1000).batch(256).prefetch(1)
test_dataset = Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(256).prefetch(1)

W = tf.Variable(
    tf.random.normal(shape=(num_features, num_classes), dtype=tf.float32), name="weight"
)
b = tf.Variable(tf.zeros(shape=(num_classes,), dtype=tf.float32), name="bias")

optimizer = SGD(learning_rate=learning_rate)


def logistic_regression(X):
    return tf.nn.softmax(tf.matmul(X, W) + b)


def cross_entropy(y_true, y_pred):
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))


def accuracy(y_true, y_pred):
    prediction = tf.equal(tf.argmax(y_pred, axis=1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(prediction, tf.float32))


def run_optimization1(x, y):
    with tf.GradientTape() as tape:
        y_pred = logistic_regression(x)
        loss = cross_entropy(y, y_pred)

    gradients = tape.gradient(target=loss, sources=[W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


for step, (x, y) in enumerate(train_dataset.take(epochs)):
    run_optimization1(x, y)

    if step % 100 == 0:
        pred = logistic_regression(x)
        loss = cross_entropy(y, pred)
        acc = accuracy(y, pred)
        print(f"step: {step:4d},  loss: {loss:.4f}, acc: {acc:.4f}")


predictions = logistic_regression(x_test)
print(f"test accuracy: {accuracy(y_test, predictions)*100:.2f}%")

num_images = 5
images = x_test[:num_images]
labels = y_test[:num_images]
predictions = logistic_regression(images)

plt.figure(figsize=(8, 4))
for i in range(num_images):
    plt.subplot(1, num_images, i + 1)
    plt.imshow(images[i].reshape(28, 28), cmap="gray")
    plt.title(f"true: {labels[i]}, pred: {np.argmax(predictions[i])}")
plt.tight_layout()
plt.show()