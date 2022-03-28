import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fonts = [(font.name, font.fname) for font in fm.fontManager.ttflist if "Nanum" in font.name]
print(fonts[0][0])
print(mpl.matplotlib_fname())

font_path = "../../data/NanumBarunGothic.ttf"
font = fm.FontProperties(fname=font_path, size=16)

plt.style.use("seaborn")
plt.rcParams["font.size"] = 16
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
tf.random.set_seed(42)

x = np.random.randn(100)
y = 2 * x + np.random.randn(100)

plt.ylabel("가격", fontproperties=font)
plt.title("가격변동 추이", fontproperties=font)
plt.plot(x, y, "b-")
plt.savefig("han_graph")

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
