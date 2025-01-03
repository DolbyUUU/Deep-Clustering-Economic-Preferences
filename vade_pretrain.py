"""
Variational Deep Embedding (VaDE) implementation
Adapted from: https://github.com/mori97/VaDE
Based on VaDE paper:
`Variational Deep Embedding An Unsupervised and Generative Approach to Clustering`
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import torch
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms
from vade_model import VaDE, AutoEncoderForPretrain


N_CLASSES = 12
DATA_DIM = 4
LATENT_DIM = 12


def train(model, data_loader, optimizer, device, epoch):
    model.train()

    total_loss = 0
    for x in data_loader:
        batch_size = x.size(0)
        x = x.to(device).view(-1, DATA_DIM)
        recon_x = model(x)
        # mse_loss vs. binary_cross_entropy
        loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {:>3}: Train Loss = {:.4f}'.format(
        epoch, total_loss / len(data_loader)))


def main():
    parser = argparse.ArgumentParser(
        description='Train VaDE with dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e',
                        help='Number of epochs.',
                        type=int, default=20)
    parser.add_argument('--gpu', '-g',
                        help='GPU id. (Negative number indicates CPU)',
                        type=int, default=-1)
    parser.add_argument('--learning-rate', '-l',
                        help='Learning Rate.',
                        type=float, default=0.001)
    parser.add_argument('--batch-size', '-b',
                        help='Batch size.',
                        type=int, default=4)
    parser.add_argument('--out', '-o',
                        help='Output path.',
                        type=str, default='./vade_parameters.pth')
    args = parser.parse_args()

    if_use_cuda = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device('cuda:{}'.format(args.gpu) if if_use_cuda else 'cpu')

    filename = 'transformed_features.txt'
    transformed_features = np.genfromtxt(filename, delimiter=' ').astype(np.float32)

    dataset = torch.from_numpy(transformed_features[:-1])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=if_use_cuda)


    pretrain_model = AutoEncoderForPretrain(data_dim=DATA_DIM, latent_dim=LATENT_DIM).to(device)

    optimizer = torch.optim.Adam(pretrain_model.parameters(),
                                 lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train(pretrain_model, data_loader, optimizer, device, epoch)

    with torch.no_grad():
        x = torch.stack([data[0] for data in dataset]).view(-1, DATA_DIM).to(device)
        z = pretrain_model.encode(x).cpu()

    pretrain_model = pretrain_model.cpu()
    state_dict = pretrain_model.state_dict()

    gmm = GaussianMixture(n_components=N_CLASSES, covariance_type="diag")
    gmm.fit_predict(z)

    model = VaDE(n_classes=N_CLASSES, data_dim=DATA_DIM, latent_dim=LATENT_DIM)
    model.load_state_dict(state_dict, strict=False)
    model._pi.data = torch.log(torch.from_numpy(gmm.weights_)).float()
    model.mu.data = torch.from_numpy(gmm.means_).float()
    model.logvar.data = torch.log(torch.from_numpy(gmm.covariances_)).float()

    torch.save(model.state_dict(), args.out)

    labels = model.classify(dataset).numpy()
    calinski_harabasz = metrics.calinski_harabasz_score(dataset, labels)
    print(labels)
    print(labels.shape)
    print(calinski_harabasz)


if __name__ == '__main__':
    main()
