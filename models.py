import numpy as np
import cvxopt
from tqdm import tqdm

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


class SVMClassifier():

    def __init__(self, C=1, kernel='rbf', gamma=0.1):
        self.C = C
        self.kernel = kernel
        if self.kernel == 'rbf':
            self.f_kernel = self.GRBF_kernel
        self.gamma = gamma

    def GRBF_kernel(self, x1, x2, gamma):
        return np.exp(-np.linalg.norm(x1-x2) * gamma)

    def fit(self, X, y, transform_y=True):
        y = y.copy()
        if transform_y:
            y = y * 2 - 1

        n, m = X.shape

        # Compute the Gram matrix
        K = np.zeros((n, n))
        for i in tqdm(range(n)):
            for j in range(n):
                K[i, j] = self.f_kernel(X[i], X[j], gamma=self.gamma)

        # construct for solver
        print(K)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n) * -1)
        A = cvxopt.matrix(y, (1, n))
        b = cvxopt.matrix(0.0)
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n) * -1))
            h = cvxopt.matrix(np.zeros(n))
        else:
            G = cvxopt.matrix(np.vstack((np.diag(np.ones(n) * -1), np.identity(n))))
            h = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        LagM = np.ravel(solution['x'])
        # Get support vectors
        self.SuppVec_indices = LagM > 1e-5
        self.supportVectors = X[self.SuppVec_indices]
        self.supportY = y[self.SuppVec_indices] * LagM[self.SuppVec_indices]

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in tqdm(range(len(y_pred))):
            s = 0
            x = X[i]
            for k in range(len(self.supportVectors)):
                s += self.f_kernel(x, self.supportVectors[k], gamma=self.gamma) * self.supportY[k]
            y_pred[i] = (s > 0) * 1

        return y_pred
