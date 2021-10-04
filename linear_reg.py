import pandas as pd
import numpy as np
import math
import random
from numpy.linalg import norm

class LinReg:
    def __init__(self, learning_rate=0.0001, tolerance=1e-2, max_iter=1e2):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.m = np.array([0.0, 0.0, 0.0])
        self.b = 0
        
    def update_weights(self, X, Y):
        m_deriv = np.array([0.0, 0.0, 0.0])
        b_deriv = 0
        N = len(X)
        for i in range(N):
            # Calculate partial derivatives w.r.t to equation (y - (mx + b))^2
            # remember we use the chain rule here.
            # -2x(y - (mx + b))

            m_deriv += -2*X[i] * (Y[i] - (self.m @ X[i] + self.b))

            # -2(y - (mx + b))
            b_deriv += -2*(Y[i] - (self.m @ X[i] + self.b))

        # We subtract because the derivatives point in direction of steepest ascent
        self.m -= (m_deriv / N) * self.learning_rate
        self.b -= (b_deriv / N) * self.learning_rate

    def cost_function(self, X, y):
        N = len(X)
        total_cost = 0
        for i in range(N):
            try:
                total_cost += math.pow((y[i] - (self.m @ X[i] + self.b)), 2)
            except OverflowError:
                return (y[i] - (self.m @ X[i] + self.b))
        return total_cost / N

    def fit(self, X, y):
        error = []
        count = 0
        while True:
            self.update_weights(X, y)
            cur_cost = self.cost_function(X, y)
            error.append(cur_cost)
            count += 1
            if cur_cost < self.tolerance:
                break
            if count > self.max_iter:
                break
            if cur_cost > 1e10:
                raise Exception("failed to converge")

    def predict(self, X):
        return np.array(
            [self.m @ X[i] + self.b for i in range(X.shape[0])]
        )

class LinRegNormed:
    def __init__(self, learning_rate=0.0001, tolerance=1e-2, max_iter=1e2, norm_coef=2):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.m = np.array([0.0, 0.0, 0.0])
        self.b = 0
        self.norm_coef = norm_coef
        
    def update_weights(self, X, Y):
        m_deriv = np.array([0.0, 0.0, 0.0])
        b_deriv = 0
        N = len(X)
        for i in range(N):
            # Calculate partial derivatives w.r.t to equation (y - (mx + b))^2
            # remember we use the chain rule here.
            # -2x(y - (mx + b))

            m_deriv += -2*X[i] * (Y[i] - (self.m @ X[i] + self.b))

            # -2(y - (mx + b))
            b_deriv += -2*(Y[i] - (self.m @ X[i] + self.b))

        # We subtract because the derivatives point in direction of steepest ascent
        self.m -= (m_deriv / N) * self.learning_rate
        self.b -= (b_deriv / N) * self.learning_rate

    def cost_function(self, X, y):
        N = len(X)
        total_cost = 0
        for i in range(N):
            try:
                total_cost += math.pow((y[i] - (self.m @ X[i] + self.b)), 2)
            except OverflowError:
                return (y[i] - (self.m @ X[i] + self.b))
        return total_cost / N

    def fit(self, X, y):
        error = []
        count = 0
        while True:
            self.update_weights(X, y)
            cur_cost = self.cost_function(X, y)
            error.append(cur_cost)
            count += 1
            if cur_cost < self.tolerance:
                break
            if count > self.max_iter:
                break
            if cur_cost > 1e10:
                raise Exception("failed to converge")

    def predict(self, X):
        return np.array(
            [self.m @ X[i] + self.b for i in range(X.shape[0])]
        )

df = pd.DataFrame()
df[0] = np.random.normal(50, 12, size=1000)
df[1] = np.random.normal(60, 14, size=1000)
df[2] = df[1] + np.random.random(size=df.shape[0])
df["y"] = 3*df[0] + 2*df[1] + 4*df[2]
df.head()
X = df[[0, 1, 2]].values
y = df["y"].values
lin_reg = LinReg()
lin_reg.fit(X, y)
print(lin_reg.predict(X))
print(lin_reg.m)
