"""
Variational Deep Embedding (VaDE) implementation
Adapted from: https://github.com/mori97/VaDE
Based on VaDE paper:
`Variational Deep Embedding An Unsupervised and Generative Approach to Clustering`
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from munkres import Munkres
from sklearn.manifold import TSNE
from sklearn import metrics
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from vade_model import VaDE, lossfun


N_CLASSES = 12
DATA_DIM = 4
LATENT_DIM = 12


def train(model, data_loader, optimizer, device, epoch, writer):
    model.train()

    total_loss = 0
    for x in data_loader:
        x = x.to(device).view(-1, DATA_DIM)
        recon_x, mu, logvar = model(x)
        loss = lossfun(model, x, recon_x, mu, logvar)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.add_scalar('Loss/train', total_loss / len(data_loader), epoch)


def main():
    parser = argparse.ArgumentParser(
        description='Train VaDE with dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e',
                        help='Number of epochs.',
                        type=int, default=50)
    parser.add_argument('--gpu', '-g',
                        help='GPU id. (Negative number indicates CPU)',
                        type=int, default=-1)
    parser.add_argument('--learning-rate', '-l',
                        help='Learning Rate.',
                        type=float, default=0.001)
    parser.add_argument('--batch-size', '-b',
                        help='Batch size.',
                        type=int, default=4)
    parser.add_argument('--pretrain', '-p',
                        help='Load parameters from pretrained model.',
                        type=str, default=None)
    args = parser.parse_args()

    if_use_cuda = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device('cuda:{}'.format(args.gpu) if if_use_cuda else 'cpu')

    filename = 'transformed_features.txt'
    transformed_features = np.genfromtxt(filename, delimiter=' ').astype(np.float32)

    dataset = torch.from_numpy(transformed_features[:-1])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=if_use_cuda)


    model = VaDE(n_classes=N_CLASSES, data_dim=DATA_DIM, latent_dim=LATENT_DIM)
    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # LR decreases every 10 epochs with a decay rate of 0.9
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.9)

    # TensorBoard
    writer = SummaryWriter('./vade_log')

    for epoch in range(1, args.epochs + 1):
        train(model, data_loader, optimizer, device, epoch, writer)
        lr_scheduler.step()

    writer.close()

    labels = model.classify(dataset).numpy()
    calinski_harabasz = metrics.calinski_harabasz_score(dataset, labels)
    print(labels)
    print(labels.shape)
    print(calinski_harabasz)


if __name__ == '__main__':
    main()
