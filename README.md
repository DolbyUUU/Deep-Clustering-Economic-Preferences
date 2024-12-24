# Economic Preferences Clustering Using Generative and Deep Learning Models

This repository contains the implementation of a research project that explores robust clustering techniques to identify economic preference types using generative and deep learning-based methods. The project uses a dataset of economic preferences and applies various clustering algorithms such as Gaussian Mixture Models (GMM), Wishart Mixture Models (WMM), and Variational Deep Embedding (VaDE) to uncover patterns in economic preferences.

---

## **Features**
- **Economic Preference Analysis**: Analyze risk aversion, time preference, and social preference.
- **Clustering Algorithms**:
  - Gaussian Mixture Models (GMM) implemented using Expectation-Maximization (EM).
  - Wishart Mixture Models (WMM) implemented using Generalized Expectation-Maximization (GEM).
  - Variational Deep Embedding (VaDE) for generative deep clustering.

---

## **Files and Descriptions**

1. `main.py`: Main source code including preprocessing, clustering, and visualization.
2. `mm_em_gaussian.py`: Implementation of Expectation-Maximization Gaussian Mixture Model (EM-GMM).
3. `mm_gem_wishart.py`: Implementation of Generalized Expectation-Maximization Wishart Mixture Model (GEM-WMM).
4. `vade_main.py`: Main script for training and evaluating VaDE.
5. `vade_pretrain.py`: Pretraining script for parameter initialization in VaDE.
6. `vade_model.py`: Implementation of the Variational Deep Embedding (VaDE) model based on the paper ["Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering"](https://arxiv.org/pdf/1611.05148.pdf). (Adapted from [mori97/VaDE](https://github.com/mori97/VaDE)).
7. `transformed_features.txt`: Dataset of transformed features after preprocessing the original data (used in VaDE).

---

## **How to Use**
Ensure you have Python 3.7.1 installed along with the following dependencies:
- `matplotlib`              3.5.2  
- `numpy`                   1.21.6  
- `scikit-learn`            0.22.1  
- `scipy`                   1.7.3  
- `tensorboard`             2.9.1  
- `tensorboardX`            2.5.1  
- `torch`                   1.2.0+cu92  
- `torchvision`             0.4.0+cu92  
