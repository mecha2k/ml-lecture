import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

font_list = [font.name for font in fontManager.ttflist if "na" in font.name]
print(font_list)

plt.style.use("seaborn")
plt.rcParams["font.size"] = 16
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "AppleGothic"  # "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
tf.random.set_seed(42)

# x = np.random.randn(100)
# y = 2 * x + np.random.randn(100)

# W = tf.Variable(np.random.randn())
# b = tf.Variable(np.random.randn())

# epochs = 1000
# learning_rate = 0.01
# optimizer = tf.optimizers.SGD(learning_rate=learning_rate)


# def linear_regression(x):
#     return x * W + b


# def mean_square(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - y_pred))


# def run_optimization():
#     with tf.GradientTape() as tape:
#         pred = linear_regression(x)
#         loss = mean_square(y, pred)
#     gradients = tape.gradient(loss, [W, b])
#     optimizer.apply_gradients(zip(gradients, [W, b]))


# for step in range(epochs):
#     run_optimization()

#     if step % 100 == 0:
#         pred = linear_regression(x)
#         loss = mean_square(y, pred)
#         print(f"Step: {step:5}\tLoss: {loss:.3f}\tW: {W.numpy():.3f}\tb: {b.numpy():.3f}")


# plt.figure(figsize=(6, 4))
# plt.plot(x, y, "bo", label="데이터")
# plt.plot(x, linear_regression(x), "r", label="Fitted 데이터")
# plt.grid()
# plt.legend()
# plt.show()
