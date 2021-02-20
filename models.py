import numpy as np
import cvxopt
from tqdm import tqdm
import scipy
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.special import expit

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
        return np.exp(-np.linalg.norm(x1 - x2) * gamma)

    # the computation of Gram matrix will be much faster using this

    def get_kernel_gram_matrix(self, X, gamma):

        if self.kernel in ['gaussian', 'rbf']:
            # Faster computation of the gram matrix with gaussian kernel
            # st= time.time()
            pairwise_dists = squareform(pdist(X, 'sqeuclidean'))
            K = np.exp(-pairwise_dists * gamma)
            # print(time.time()-st)
            return K

    def fit(self, X, y, transform_y=True):
        y = y.copy()
        if transform_y:
            y = y * 2 - 1

        n, m = X.shape

        # the computation of Gram matrix will be much faster using this
        K = self.get_kernel_gram_matrix(X, self.gamma)

        '''K1 = np.zeros((n,n))
        for i in tqdm(range(n)):
            for j in range(n):
                K1[i, j] = self.f_kernel(X[i], X[j], gamma=self.gamma)'''


        # construct for solver
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

    def predict_probas(self, X):

        try:
            assert self.kernel in ['gaussian', 'rbf']
            # compute pairwise (squared euclidean) distances between new samples and support vectors
            pairwise_dists = cdist(self.supportVectors, X, 'sqeuclidean')
            # gaussian kernel evaluations
            K_pred = np.exp(-pairwise_dists * self.gamma)
            
            #fixed error: due to self.weight instead of self.supportY
            pred_probas = expit(K_pred.T @ self.supportY)
            return pred_probas

        except:
            print('Please make sure the used kernel is gaussian.')

    def predict(self, X):
        probas = self.predict_probas(X=X)
        return (probas > 0.5).astype(int)

    def get_accuracy_score(self, X, y):
        pred_labels = self.predict(X=X)
        return (pred_labels == y).mean()


class WKRR():
    """
    Weighted Kernel Ridge Regression

    """

    def __init__(self, kernel='gaussian'):
        self.weights = None
        self.kernel = kernel
        # kernel gram matrix over training data
        self.K_train = None
        # training samples
        self.X_train = None

    def get_kernel_gram_matrix(self, X, sigma):

        if self.kernel == 'gaussian':
            # Faster computation of the gram matrix with gaussian kernel
            # st= time.time()
            pairwise_dists = squareform(pdist(X, 'sqeuclidean'))
            K = np.exp(-pairwise_dists / (2 * np.square(sigma)))
            # print(time.time()-st)
            return K

    def fit(self, X, y, penalty, W=None, eps=1e-6, kernel_precomputed=False):
        """
        Returns analytical solution of the Weighted Kernel Ridge Regression problem
        """

        self.X_train = X
        if kernel_precomputed:
            K = self.K_train
        else:
            K = self.get_kernel_gram_matrix(X, self.sigma)
            self.K_train = K

        assert K.shape[0] == y.shape[0]
        n = K.shape[0]

        if W is None:
            # unweighted KLR := all weights are equal to 1 and W:=Identity
            M = K @ y + n * penalty * np.eye(n)
            M_ = scipy.linalg.inv(M)
            v = M_ @ v

        else:

            W_sqrt = np.diag(np.sqrt(np.diag(W)))
            v = W_sqrt @ y
            M = K @ W_sqrt
            M = W_sqrt @ M + n * penalty * np.eye(n)
            M_ = scipy.linalg.inv(M)
            v = M_ @ v
            v = W_sqrt @ v

        print('fitted train data')
        self.weights = v


class KernelLogisticRegression():
    """
    Kernel Logistic regression
    """

    def __init__(self, kernel='gaussian', sigma=1):
        self.weights = None
        self.kernel = 'kernel'
        self.sigma = sigma
        self.loss_thresh = 0.001
        self.X_train = None

        # initialize weighted kernel ridge regression for self.fit
        self.wkrr = WKRR()

    def get_kernel_gram_matrix(self, X, sigma):

        if self.kernel == 'gaussian':
            # Faster computation of the gram matrix with gaussian kernel
            # st= time.time()
            pairwise_dists = squareform(pdist(X, 'sqeuclidean'))
            K = np.exp(-pairwise_dists / (2 * np.square(sigma)))
            # print(time.time()-st)
            return K

    def fit(self, X, y, penalty, max_iter=1000, eps=1e-6):
        """
        Iteratively solve Weighted Kernel Ridge Regression problems
        """

        self.X_train = X
        K = self.get_kernel_gram_matrix(X, self.sigma)
        self.wkrr.K_train = K

        # For training only, transform labels in [1,-1]
        y = np.where(y == 1, 1, 0)

        assert K.shape[0] == y.shape[0]
        n = K.shape[0]
        ones = np.ones(y.shape)

        # randomly initialize the coefficients
        alpha = np.random.normal(loc=0, scale=1, size=n)

        # t1 = time.time()

        # initialize loss
        loss = 10

        for i in range(max_iter):
            # At each iteration solve a Weighted kernel ridge regression
            v = K @ alpha
            prev_loss = loss
            loss = -np.sum(np.log(expit(np.multiply(y, v)) + eps)) / n
            print(loss)
            if np.abs(loss - prev_loss) < self.loss_thresh:
                print('converged after {} iterations'.format(i + 1))
                break

            # compute parameters for WKRR
            u = np.multiply(v, y)
            sig = expit(u)
            sig_ = ones - sig  # 1-sig = expit(-u)
            W = np.diag(np.multiply(sig, (sig_)))
            # print(W.shape)
            P = np.diag(-sig_)
            k = P @ y
            z = v - scipy.linalg.inv(W) @ k
            # t2 = time.time()
            # print(t2-t1)

            # solve a weighted Kernel Ridgre Regression with the corresponding parameters
            alpha = self.wkrr.fit(X, z, penalty, sigma, W=W, kernel_precomputed=True)
            # t3 = time.time()
            # print(t3-t2)

        # save fitted parameters
        self.weights = alpha

    def predict_probas(self, X):

        try:
            assert self.kernel == 'gaussian'
            # compute pairwise (squared euclidean) distances between new samples and train samples
            pairwise_dists = cdist(self.X_train, X, 'sqeuclidean')
            # gaussian kernel evaluations
            K_pred = np.exp(-pairwise_dists / (2 * np.square(self.sigma)))

            pred_probas = expit(K_pred.T @ self.weights)
            return pred_probas

        except:
            print('Please make sure the used kernel is gaussian.')

    def predict(self, X):
        probas = self.predict_probas(X=X)
        return (probas > 0.5).astype(int)

    def get_accuracy_score(self, X, y):
        pred_labels = self.predict(X=X)
        return (pred_labels == y).mean()
