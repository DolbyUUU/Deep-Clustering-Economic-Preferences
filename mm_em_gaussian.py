import numpy as np
from scipy.stats import multivariate_normal

class GaussianMM():
    def __init__(self, k, dim, max_iter=100, n_init=100, regul=1e-6, init_mu=None, init_sigma=None, init_pi=None):
        '''
        Define a Gaussian mixture model with known number of clusters and dimensions.
        input:
            - k: number of Gaussian clusters or components
            - dim: dimension or number of features
            - max_iter: maximum number of EM iterations to perform
                default = 100
            - n_init: number of initializations to perform and the best results are kept
                default = 100
            - regul: non-negative regularization added to the diagonal of covariance matrices
                default = 1e-6
            - init_mu: initial value of mean of clusters (k, dim)
                (default) random from uniform[-10, 10]
            - init_sigma: initial value of covariance matrix of clusters (k, dim, dim)
                (default) identity matrix for each cluster
            - init_pi: initial value of cluster weights (k,)
                (default) equal value to all cluster, i.e., 1/k
        '''
        self.k = k
        self.dim = dim
        self.max_iter = max_iter
        self.n_init = n_init
        self.regul = regul
        if (init_mu is None):
            # Random initialization of Gaussian mean mu with uniform[-10, 10].
            init_mu = np.random.rand(k, dim) * 20 - 10
        self.mu = init_mu
        if (init_sigma is None):
            init_sigma = np.zeros((k, dim, dim))
            for i in range(k):
                init_sigma[i] = np.identity(dim)
        self.sigma = init_sigma
        if (init_pi is None):
            init_pi = np.ones(self.k) / self.k
        self.pi = init_pi

    def init_em(self, X):
        '''
        Initialization for EM algorithm.
        input:
            - X: data (batch_size, dim)
        '''
        self.data = X
        self.num_points = X.shape[0]
        self.z = np.zeros((self.num_points, self.k))

    def e_step(self):
        '''
        E-step of EM algorithm.
        '''
        for j in range(self.k):
            self.z[:, j] = self.pi[j] * multivariate_normal.pdf(self.data, mean=self.mu[j], cov=self.sigma[j])
        self.z /= self.z.sum(axis=1, keepdims=True)

    def m_step(self):
        '''
        M-step of EM algorithm.
        '''
        sum_z = self.z.sum(axis=0)
        self.pi = sum_z / self.num_points
        self.mu = np.matmul(self.z.T, self.data)
        self.mu /= sum_z[:, None]
        for j in range(self.k):
            minus = np.expand_dims(self.data, axis=1) - self.mu[j]
            squared = np.matmul(minus.transpose([0, 2, 1]), minus)
            self.sigma[j] = np.matmul(squared.transpose(1, 2, 0), self.z[:, j])
            self.sigma[j] /= sum_z[j]
            # Regularization of the covariance matrices sigma_k.
            self.sigma[j] += np.identity(self.dim) * self.regul

    def log_likelihood(self, X):
        '''
        Compute the log-likelihood of X under current parameters.
        input:
            - X: data (batch_size, dim)
        output:
            - log-likelihood of X: Sum_n Sum_k log(pi_k * N( X_n | mu_k, sigma_k ))
        '''
        ll = []
        for d in X:
            tot = 0
            for j in range(self.k):
                tot += self.pi[j] * multivariate_normal.pdf(d, mean=self.mu[j], cov=self.sigma[j])
            ll.append(np.log(tot))
        data_ll = np.sum(ll)
        return data_ll

    def bic_aic(self, X):
        '''
        Compute BIC and AIC of X under current parameters.
        input:
            - X: data (batch_size, dim)
        output:
            - bic (Bayesian information criterion): -2 * log_L + log_N * num_param
            - aic (Akaike information criterion): -2 * log_L + 2 * num_param
        '''
        data_ll = self.log_likelihood(X)
        dof = (self.dim * self.dim - self.dim) / 2 + 2 * self.dim + 1
        num_param = self.k * dof - 1
        bic = -2 * data_ll + np.log(self.num_points) * num_param
        aic = -2 * data_ll + 2 * num_param
        return bic, aic

    def fit(self, X):
        '''
        Train the model with input data.
        input:
            - X: data (batch_size, dim)
        '''
        self.init_em(X)
        for i in range(self.max_iter):
            self.e_step()
            self.m_step()

    def predict(self, X):
        '''
        Predict the cluster assignments of input data.
        input:
            - X: data (batch_size, dim)
        output:
            - cluster_labels: list of cluster assignments, i.e., labels of X.
            - cluster_centers: list of cluster centroids, i.e., means of Gaussian clusters.
            - cluster_weights: list of cluster weights.
        '''
        probas = []
        cluster_labels = []
        cluster_centers = []
        cluster_weights = []
        for i in range(len(X)):
            probas.append([multivariate_normal.pdf(X[i], mean=self.mu[j], cov=self.sigma[j]) for j in range(self.k)])
        for proba in probas:
            cluster_labels.append(proba.index(max(proba)))
        for j in range(self.k):
            cluster_centers.append(self.mu[j])
        for j in range(self.k):
            cluster_weights.append(self.pi[j])
        return np.array(cluster_labels), np.array(cluster_centers), np.array(cluster_weights)

    def trial(self, X):
        '''
        Initialize randomly and select trial with largest data log-likelihood.
        input:
            - X: data (batch_size, dim)
        output:
            - largest log-likelihood of X in n_init trials.
            - sample labels of the best trial.
            - cluster centers of the best trial.
        '''
        i_iter = 0
        data_ll_lst, bic_lst, aic_lst, cluster_labels_lst, cluster_centers_lst, cluster_weights_lst = \
        [], [], [], [], [], []
        while self.n_init >= i_iter:
            i_iter += 1
            print(f"Initialization Trial No: {i_iter}")
            self.fit(X)
            cluster_labels, cluster_centers, cluster_weights = self.predict(X)
            data_ll = self.log_likelihood(X)
            bic, aic = self.bic_aic(X)
            data_ll_lst.append(data_ll)
            bic_lst.append(bic)
            aic_lst.append(aic)
            cluster_labels_lst.append(cluster_labels)
            cluster_centers_lst.append(cluster_centers)
            cluster_weights_lst.append(cluster_weights)
        for i in range(len(data_ll_lst)):
            if data_ll_lst[i] == max(data_ll_lst):
                best_data_ll = data_ll_lst[i]
                best_bic = bic_lst[i]
                best_aic = aic_lst[i]
                best_cluster_labels = cluster_labels_lst[i]
                best_cluster_centers = cluster_centers_lst[i]
                best_cluster_weights = cluster_weights_lst[i]
        print(f"Best result in {self.n_init} random trials:")
        print(f"\nData log-likelihood: {best_data_ll}")
        print(f"\nBIC: {best_bic}")
        print(f"\nAIC: {best_aic}")
        print(f"\nCluster centers: {best_cluster_centers}")
        print(f"\nCluster weights: {best_cluster_weights}")
        return best_cluster_labels, best_cluster_centers, best_cluster_weights, best_data_ll, best_bic, best_aic
