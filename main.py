import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
from scipy import linalg
from sklearn import impute, preprocessing, decomposition, cluster, mixture, metrics, manifold
from matplotlib.ticker import MaxNLocator
from pyclustertend import hopkins
from mm_em_gaussian import GaussianMM
from mm_gem_wishart import WishartMM


# Import dataset.
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data_guided_project.csv')
df = pd.read_csv(filename, index_col=0)


# Remove features:
# (1) subject age
# (2) experimental group
# (3) the risk preference dimension with many missing values.
df_features = df.drop(columns=["age", "group", "risk_seeking_5"])


# Remove samples with 3 or more missing values.
df_features = df_features[df_features.isnull().sum(axis=1) < 3]


# Split dataframe into four sub-dataframes representing feature groups of risk, ambiguity, time and social preferences.
df_risk_features = df_features.iloc[:, 2:7]
df_ambiguity_features = df_features.iloc[:, 7:13]
df_time_features = df_features.iloc[:, 13:15]
df_sociality_features = df_features.iloc[:, 15:]

print(f"Dim of risk features: {df_risk_features.shape[1]}")
print(f"Dim of ambiguity features: {df_ambiguity_features.shape[1]}")
print(f"Dim of time features: {df_time_features.shape[1]}")
print(f"Dim of sociality features: {df_sociality_features.shape[1]}")


feature_list = [df_risk_features, df_ambiguity_features, df_time_features, df_sociality_features]


# Impute missing values with nearest neighbors imputation for each sub-dataframe.
imputer = impute.KNNImputer(
    missing_values=np.nan, n_neighbors=5, weights="uniform", metric="nan_euclidean", copy=False
    )
for feature in feature_list:
    imputer.fit_transform(feature)


# Use PCA to reduce dimensionality and extract the first PC for each sub-dataframe.
def pre_processing(dataframe):
    """
    Preprocess features before PCA.
    """

    if standardize == True:
        # Standardize features by removing the mean and scaling to unit variance.
        scaler = preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(dataframe)

    if normalize == True:
        # # Normalize features by scaling each feature to the range between 0 and 1.
        scaler = preprocessing.MinMaxScaler(copy=False)
        scaler.fit_transform(dataframe)

    return dataframe


standardize = True
normalize = False

for feature in feature_list:
    pre_processing(feature)


def kernel_pc_analysis(dataframe, title):
    """
    Conduct kernel PCA and plot explained variance ratio vs. no. of PCs.
    """
    global kpca

    # Plot explained variance ratio vs. no. of PCs.
    kpca = decomposition.KernelPCA(n_components=dataframe.shape[1], kernel="poly")
    kpca_transform = kpca.fit_transform(dataframe)
    explained_variance = np.var(kpca_transform, axis=0)
    exp_var_pca = explained_variance / np.sum(explained_variance)
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    fig, ax = plt.subplots()
    plt.bar(range(1, len(exp_var_pca)+1), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, len(cum_sum_eigenvalues)+1), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='upper left')
    plt.title(title)
    # plt.savefig(title.replace(".", "_").replace(" ", "_").replace(",", "").replace("=", "").replace("(", "").replace(")", "").lower())
    plt.close()

    # Extract the first PC as the new feature.
    kpca = decomposition.KernelPCA(n_components=1, kernel="poly")
    kpca.fit(dataframe)


def pc_analysis(dataframe, title):
    """
    Conduct PCA and plot explained variance ratio vs. no. of PCs.
    """
    global pca

    # Plot explained variance ratio vs. no. of PCs.
    pca = decomposition.PCA(n_components=dataframe.shape[1])
    pca.fit(dataframe)
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    fig, ax = plt.subplots()
    plt.bar(range(1, len(exp_var_pca)+1), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, len(cum_sum_eigenvalues)+1), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='upper left')
    plt.title(title)
    # plt.savefig(title.replace(".", "_").replace(" ", "_").replace(",", "").replace("=", "").replace("(", "").replace(")", "").lower())
    plt.close()

    # Extract the first PC as the new feature.
    pca = decomposition.PCA(n_components=1)
    pca.fit(dataframe)


pca = True
kernel = True

if pca == True:
    if kernel == True:
        kernel_pc_analysis(df_risk_features, "Kernel PCA of risk preference features")
        risk_pc = kpca.transform(df_risk_features)
        kernel_pc_analysis(df_ambiguity_features, "Kernel PCA of ambiguity preference features")
        ambiguity_pc = kpca.transform(df_ambiguity_features)
        kernel_pc_analysis(df_time_features, "Kernel PCA of time preference features")
        time_pc = kpca.transform(df_time_features)
        kernel_pc_analysis(df_sociality_features, "Kernel PCA of social preference features")
        sociality_pc = kpca.transform(df_sociality_features)
    else:
        pc_analysis(df_risk_features, "PCA of risk preference features")
        risk_pc = pca.transform(df_risk_features)
        pc_analysis(df_ambiguity_features, "PCA of ambiguity preference features")
        ambiguity_pc = pca.transform(df_ambiguity_features)
        pc_analysis(df_time_features, "PCA of time preference features")
        time_pc = pca.transform(df_time_features)
        pc_analysis(df_sociality_features, "PCA of social preference features")
        sociality_pc = pca.transform(df_sociality_features)
else:
    risk_pc = df_risk_features
    ambiguity_pc = df_ambiguity_features
    time_pc = df_time_features
    sociality_pc = df_sociality_features


# Concatenate four columns of the first PCs into one dataframe.
transformed_features = np.concatenate((risk_pc, ambiguity_pc, time_pc, sociality_pc), axis=1)


# Scale features before using hopkins or vat algorithm as they use distance between observations.

scaler = preprocessing.StandardScaler(copy=False)
scaler.fit_transform(transformed_features)

# Output transformed_features to txt.
np.savetxt("transformed_features.txt", transformed_features)

# Compute Hopkins statistic to assess the clusterability.
sampling_ratio_list = [0.05, 0.1, 0.15, 0.2]

for sampling_ratio in sampling_ratio_list:
    sample_size = round(sampling_ratio * transformed_features.shape[0])
    hopkins_score = round(hopkins(transformed_features, sample_size), 2)
    print(f"Hopkins: sampling ratio {sampling_ratio}, sampling size {sample_size}, score {hopkins_score}")


# Gaussian mixture model selection.
cov_type_lst = ['diag', 'full', 'spherical', 'tied']
threshold_lst = [1, 0.975, 0.95, 0.90]
for threshold in threshold_lst:
    lowest_bic = np.infty
    bic_lst = []
    ll_lst = []
    for num_clusters in range(1, 21):
        for cov_type in cov_type_lst:
            gmm = mixture.GaussianMixture(n_components=num_clusters, covariance_type=cov_type)
            gmm.fit(transformed_features)
            bic_lst.append(gmm.bic(transformed_features))
            ll_lst.append(gmm.score(transformed_features))
            if bic_lst[-1] < threshold * lowest_bic: # GMM with fewer clusters is prefered.
                lowest_bic = bic_lst[-1]
                best_ll = ll_lst[-1]
                best_cov_type = cov_type
                best_num_clusters = num_clusters
                best_gmm = gmm
    print(f"At threshold {threshold}, GMM with lowest BIC is {best_gmm}"
        f"\nBIC = {lowest_bic}"
        f"\nData LL = {best_ll}")


# Bayesian Gaussian mixture model selection.
# BGMM has no BIC or AIC.
cov_type_lst = ['diag', 'full', 'spherical', 'tied']
min_weight_lst = [0.01, 0.025, 0.05, 0.075, 0.10]
for min_weight in min_weight_lst:
    for cov_type in cov_type_lst:
        bgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type=cov_type)
        bgmm.fit(transformed_features)
        ll = bgmm.score(transformed_features)
        cluster_weights = bgmm.weights_
        num_effective_clusters = sum(i > min_weight for i in cluster_weights)
        print(f"BGMM: {bgmm}"
        f"\nComponent weight threshold = {min_weight}"
        f"\nNumber of effective components = {num_effective_clusters}"
        f"\nData LL = {ll}")


NUM_CLUSTERS = 12


def tsne_plot(perplexity, labels, num_clusters, title):
    tsne = manifold.TSNE(n_components=2, learning_rate="auto", perplexity=perplexity, init="random", random_state=0)
    tsne_transformed_features = tsne.fit_transform(transformed_features)
    df = pd.DataFrame()
    df["y"] = labels
    df["component_1"] = tsne_transformed_features[:, 0]
    df["component_2"] = tsne_transformed_features[:, 1]
    tsne_plot = sns.scatterplot(x="component_1", y="component_2", hue=df.y.tolist(),
        palette=sns.color_palette("hls", num_clusters), data=df).set(title=title)
    plt.savefig(title.replace(".", "_").replace(" ", "_").replace(",", "")
        .replace("=", "").replace("(", "").replace(")", "").lower())
    plt.close()


# Wishart mixture model (generalized EM).
wmm = WishartMM(k=NUM_CLUSTERS, dim=4, size=1000, max_iter=100, n_init=0)
cluster_labels, cluster_weights, data_ll, bic, aic = wmm.trial(transformed_features)
data_ll /= len(transformed_features)
calinski_harabasz = metrics.calinski_harabasz_score(transformed_features, cluster_labels)
print(f"GEM-WMM number of clusters = {NUM_CLUSTERS}"
    f"\nBIC = {bic}"
    f"\nData LL = {data_ll}"
    f"\nCalinski Harabasz Index = {calinski_harabasz}"
for perplexity in [10, 20, 30, 40, 50]:
    tsne_plot(perplexity=perplexity, labels=labels, num_clusters=NUM_CLUSTERS,
    title=f"T-SNE visualization of WMM clustering, perplexity={perplexity}")


best_gmm = mixture.GaussianMixture(n_components=NUM_CLUSTERS, covariance_type='diag')
labels = best_gmm.fit_predict(transformed_features)
calinski_harabasz = metrics.calinski_harabasz_score(transformed_features, labels)

print(f"GMM with number of clusters = {NUM_CLUSTERS}"
    f"\nBIC = {best_gmm.bic(transformed_features)}"
    f"\nData LL = {best_gmm.score(transformed_features)}"
    f"\nCluster weights = {best_gmm.weights_}"
    f"\nCluster centroids = {best_gmm.means_}"
    f"\nCalinski Harabasz Index = {calinski_harabasz}"
for perplexity in [10, 20, 30, 40, 50]:
    tsne_plot(perplexity=perplexity, labels=labels, num_clusters=NUM_CLUSTERS,
    title=f"T-SNE visualization of GMM clustering, perplexity={perplexity}")


best_bgmm = mixture.BayesianGaussianMixture(n_components=NUM_CLUSTERS, covariance_type='diag')
labels = best_bgmm.fit_predict(transformed_features)
calinski_harabasz = metrics.calinski_harabasz_score(transformed_features, labels)

print(f"BGMM with number of clusters = {NUM_CLUSTERS}"
    f"\nData LL = {best_bgmm.score(transformed_features)}"
    f"\nCluster weights = {best_bgmm.weights_}"
    f"\nCluster centroids = {best_bgmm.means_}"
    f"\nCalinski Harabasz Index = {calinski_harabasz}"

for perplexity in [10, 20, 30, 40, 50]:
    tsne_plot(perplexity=perplexity, labels=labels, num_clusters=NUM_CLUSTERS,
    title=f"T-SNE visualization of BGMM clustering, perplexity={perplexity}")
