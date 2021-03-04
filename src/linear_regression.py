import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def linearRegression():
    X = 10 * np.random.random(100)
    y = 2 * X + 1 + 3 * np.random.rand(100)

    linRegression = LinearRegression()
    linRegression.fit(X.reshape(-1, 1), y)
    y_pred = linRegression.predict(X.reshape(-1, 1))
    print(linRegression.coef_[0], linRegression.intercept_)

    plt.plot(X, y, "o")
    plt.plot(X, y_pred)
    plt.show()


if __name__ == "__main__":
    linearRegression()
