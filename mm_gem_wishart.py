import numpy as np
from scipy.stats import wishart

class WishartMM():
    def __init__(self, k, dim, size=200, max_iter=100, n_init=100, init_df=None, init_scale=None, init_pi=None):
        '''
        Define a Wishart mixture model with known number of clusters and dimensions.
        input:
            - k: number of Wishart clusters or components
            - dim: dimension or number of features
            - size: batch size or sample size
                default = 200
            - max_iter: maximum number of EM iterations to perform
                default = 100
            - n_init: number of initializations to perform and the best results are kept
                default = 100
            - init_df: initial value of degrees of freedom in integers, >= dim of scale matrix (k,)
                (default) random from uniform[dim, dim + 0.5 * size]
            - init_scale: initial value of symmetric positive definite scale matrix (k, dim, dim)
                (default) identity matrix for each cluster
            - init_pi: initial value of cluster weights (k,)
                (default) equal value to all cluster, i.e., 1/k
        '''
        self.k = k
        self.dim = dim
        self.max_iter = max_iter
        self.n_init = n_init
        if (init_df is None):
            # Random initialization of Wishart degrees of freedom df with uniform[dim, dim + 0.5 * size].
            init_df = np.random.rand(k) * size * 0.5 + dim
            init_df = np.intc(np.rint(init_df))
        self.df = init_df
        if (init_scale is None):
            init_scale = np.zeros((k, dim, dim))
            # Standard Wishart distribution, i.e., scale matrix = identity matrix.
            for i in range(k):
                init_scale[i] = np.identity(dim)
        self.scale = init_scale
        if (init_pi is None):
            init_pi = np.ones(self.k) / self.k
        self.pi = init_pi

    def init_em(self, X):
        '''
        Initialization for generalized EM algorithm.
        input:
            - X: data (batch_size, dim)
        '''
        self.data = X
        self.data_descriptor = np.matmul(X.T, X)
        self.num_points = X.shape[0]
        self.z = np.zeros((self.num_points, self.k))

    def e_step(self):
        '''
        E-step of generalized EM algorithm.
        '''
        for j in range(self.k):
            self.z[:, j] = self.pi[j] * wishart.pdf(self.data_descriptor, df=self.df[j], scale=self.scale[j])
        self.z /= self.z.sum(axis=1, keepdims=True) + 1e-6

    def m_step(self):
        '''
        M-step of generalized EM algorithm.
        '''
        data_ll = self.log_likelihood(self.data)
        sum_z = self.z.sum(axis=0)
        self.pi = sum_z / self.num_points
        self.mu = np.matmul(self.z.T, self.data)
        self.mu /= sum_z[:, None] + 1e-6
        # Random search for degrees of freedom df that can increase the log-likelihood of X.
        temp_df = np.intc(np.rint(np.random.rand(self.k) * self.df * 2) + self.dim)
        temp_ll = []
        for d in self.data:
            data_descriptor = np.matmul(np.transpose(d), d)
            tot = 0
            for j in range(self.k):
                tot += self.pi[j] * wishart.pdf(data_descriptor, df=temp_df[j], scale=self.scale[j])
            temp_ll.append(np.log(tot + 1e-6))
        temp_data_ll = np.sum(temp_ll)
        if temp_data_ll > data_ll:
            self.df = temp_df

    def log_likelihood(self, X):
        '''
        Compute the log-likelihood of X under current parameters.
        input:
            - X: data (batch_size, dim)
        output:
            - log-likelihood of X: Sum_n Sum_k log(pi_k * Wishart( X_n | df_k, scale_k ))
        '''
        ll = []
        for d in X:
            data_descriptor = np.matmul(np.transpose(d), d)
            tot = 0
            for j in range(self.k):
                tot += self.pi[j] * wishart.pdf(data_descriptor, df=self.df[j], scale=self.scale[j])
            ll.append(np.log(tot + 1e-6))
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
            - cluster_centers: list of cluster centroids, i.e., means of Wishart clusters.
            - cluster_weights: list of cluster weights.
        '''
        probas = []
        cluster_labels = []
        cluster_centers = []
        cluster_weights = []
        for i in range(len(X)):
            data_descriptor = np.matmul(np.transpose(X[i]), X[i])
            probas.append([wishart.pdf(data_descriptor, df=self.df[j], scale=self.scale[j]) for j in range(self.k)])
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
        print(f"\nDegrees of freedom: {self.df}")
        print(f"\nCluster weights: {best_cluster_weights}")
        return best_cluster_labels, best_cluster_weights, best_data_ll, best_bic, best_aic
