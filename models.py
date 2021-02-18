import numpy as np


class LogisticRegression():

    def __init__(self):
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_probas(self, X):
        z = X @ self.weights
        return self.sigmoid(z)

    def cross_entropy(self, X, y):
        n = X.shape[0]
        y_pred = self.predict_probas(X, self.weights)
        cost = y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        return cost.sum() / n

    def get_grad(self, X, y):
        n = X.shape[0]
        y_pred = self.predict_probas(X=X)

        return X.T @ (y_pred - y) / n

    def fit(self, X, y, max_iter=1000, lr=1, eps=1e-6):
        self.weights = np.zeros(X.shape[1])
        cv = False
        j = 1
        for i in range(max_iter):
            weights_prev = self.weights.copy()
            grad = self.get_grad(X, y)
            self.weights -= lr * grad
            if np.linalg.norm(self.weights - weights_prev, 2) < eps:
                cv =True
                print('Algorithm converged !')
                break
            if (i/10000 == j):
                lr /= 2
                j += 1
        if not(cv):
            print('Reached maximum iterations without convergence.')

    def predict(self, X):
        probas = self.predict_probas(X=X)
        return (probas>0.5).astype(int)

    def get_accuracy_score(self, X, y):
        pred_labels = self.predict(X=X)
        return (pred_labels==y).mean()