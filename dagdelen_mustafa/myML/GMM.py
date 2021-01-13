import numpy as np
from .KMeans import KMeans


class GMM:
    def __init__(self, n_clusters=3, max_iteration=100, tol=1e-3, init_params="random", reg_val=1e-6):
        self.n_clusters = n_clusters
        self.max_iteration = max_iteration
        self.tol = tol
        self.init_params = init_params
        self.reg_val = reg_val

    def initialize_params(self, X):
        if self.init_params == "kmeans":
            self.r = np.zeros((X.shape[0], self.n_clusters))
            kmeans = KMeans(n_clusters=self.n_clusters)
            kmeans.fit(X)
            self.r[np.arange(X.shape[0]), kmeans.labels] = 1
            self.N = np.sum(self.r, axis=0)
            self.means_ = kmeans.cluster_centers_
            self.covariances = self.calculate_covariances(X)

        elif self.init_params == "random":
            self.r = np.random.rand(X.shape[0], self.n_clusters)
            self.r /= self.r.sum(axis=1)[:, np.newaxis]
            self.calculate_params(X)
        else:
            raise ValueError("Unimplemented initialization method '%s'" % self.init_params)

        self.pi = np.random.dirichlet(np.ones(self.n_clusters), size=1)[0]

    def calculate_params(self, X):
        self.N = np.sum(self.r, axis=0)
        self.means_ = (self.r.T @ X) / self.N[:, np.newaxis]
        self.covariances = self.calculate_covariances(X)
        self.pi = self.N / self.n_clusters

    def calculate_covariances(self, X):
        n_features = self.means_.shape[1]
        covariances = np.empty((self.n_clusters, n_features, n_features))
        for k in range(self.n_clusters):
            diff = X - self.means_[k]
            covariances[k] = np.dot(self.r[:, k] * diff.T, diff) / self.N[k]
            covariances[k] += self.reg_val
        return covariances

    def mnd(self, X, mean, covariance):
        d = X.shape[0]
        diff = X - mean
        numerator = np.exp(-np.dot(np.dot(diff.T, np.linalg.inv(covariance)), diff) / 2)
        denominator = np.power((2 * np.pi), (d / 2)) * np.power(np.linalg.det(covariance), 0.5)
        return numerator / denominator

    def e_step(self, X):
        r = np.zeros((X.shape[0], self.n_clusters))
        for i in range(X.shape[0]):
            normalization = sum(
                self.pi[k] * self.mnd(X[i], self.means_[k], self.covariances[k]) for k in range(self.n_clusters))
            for c in range(self.n_clusters):
                r[i][c] = self.pi[c] * self.mnd(X[i], self.means_[c], self.covariances[c]) / normalization
        self.r = r

    def m_step(self, X):
        self.calculate_params(X)

    def is_converged(self, p_means):
        diff = np.sum(np.abs(self.means_ - p_means) / (p_means * 100.0))
        return diff < self.tol

    def fit(self, X):
        self.initialize_params(X)

        for it in range(self.max_iteration):
            self.e_step(X)
            p_means = self.means_.copy()
            self.m_step(X)
            if self.is_converged(p_means):
                print("Final iteration: {}".format(it))
                break
            if it % 20 == 0:
                print("Iteration: {}".format(it + 1))

    def predict(self, X):
        return np.argmax(np.array(
            [[self.mnd(X[n], self.means_[k], self.covariances[k]) for k in range(self.n_clusters)] for n in
             range(X.shape[0])]), axis=1)

