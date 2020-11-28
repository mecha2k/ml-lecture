import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# fmt: off
x_data = [
    [73.0, 80.0, 75.0],
    [93.0, 88.0, 93.0],
    [89.0, 91.0, 90.0],
    [96.0, 98.0, 100.0],
    [73.0, 66.0, 70.0]
]
y_data = [
    [152.0],
    [185.0],
    [180.0],
    [196.0],
    [142.0]
]
# fmt: on

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 1000 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
