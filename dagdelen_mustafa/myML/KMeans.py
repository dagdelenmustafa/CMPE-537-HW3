import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, tol=1e-3, max_iter=200, centers=None):
        self.k = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.cluster_centers_ = centers

    def fit(self, data):
        if self.cluster_centers_ is None:
            self.cluster_centers_ = self.generate_kmeans_pp_centers(data, self.k)

        for i in range(self.max_iter):
            self.clusters = {}
            dist_matrix = self.euclidean_distance(data, self.cluster_centers_)
            self.closest_cluster_ids = np.argmin(dist_matrix, axis=1)

            for k in range(self.k):
                self.labels = []
                self.clusters[k] = []

            for j, id in enumerate(self.closest_cluster_ids):
                self.clusters[id].append(data[j])
                self.labels.append(id)

            prev_centers = self.cluster_centers_.copy()
            self.cluster_centers_ = np.array([np.mean(self.clusters[key], axis=0, dtype=data.dtype) for key in range(self.k)])

            distances_between_centers = self.euclidean_distance(prev_centers, self.cluster_centers_)
            max_dist = np.max(distances_between_centers.diagonal())
            is_covered = max_dist <= self.tol

            if i % 10 == 0:
                print("Iteration: {}, Distance: {}".format(i, max_dist))

            if is_covered:
                print("Max iteration: ".format(i))
                break

    def euclidean_distance(self, first, second):
        first_square = np.reshape(np.sum(first ** 2, axis=1), (first.shape[0], 1))
        second_square = np.reshape(np.sum(second ** 2, axis=1), (1, second.shape[0]))
        dist = (-2 * first.dot(second.T)) + second_square + first_square

        return np.sqrt(dist)

    def generate_kmeans_pp_centers(self, data, k):
        """
        K-Means++ initialization method was used because of the initialization error
        which was mentioned in 'https://datascience.stackexchange.com/q/44897'

        Source: https://www.kdnuggets.com/2020/06/centroid-initialization-k-means-clustering.html
        """
        centroids = [data[0]]

        for _ in range(1, k):
            dist_sq = np.array([min([np.inner(c - x, c - x) for c in centroids]) for x in data])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break

            centroids.append(data[i])

        return np.array(centroids)
