import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import datasets, transforms

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F


def hello():
    print('Hello from cvae.py!')


class CVAE(nn.Module):

    def __init__(self, *,
                 input_size, latent_size, num_classes, hidden_units=400):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.latent_size = latent_size
        self.units = hidden_units

        #######################################################
        ###              START OF YOUR CODE                 ###
        #######################################################
        ### Define a three layer neural network architecture
        ### for the recognition_model. You are free to choose
        ### any reasonable architecture and implementation
        ### details that would make CVAE work.
        #######################################################
        # ~~START DELETE~~
        self.hidden_dim = self.units

        # Encoder
        layers = []
        layers.append(nn.Linear(self.input_size + self.num_classes, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

        self.mu_layer = nn.Linear(self.hidden_dim + self.num_classes, self.latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim + self.num_classes, self.latent_size)

        # ~~END DELETE~~
        #######################################################
        ###               END OF YOUR CODE                  ###
        #######################################################


        #######################################################
        ###              START OF YOUR CODE                 ###
        #######################################################
        # Define a three layer neural network architecture for
        # the generation_model. You are free to choose
        # architectures and layers as you wish.

        # ~~START DELETE~~
        # Decoder
        layers2 = []
        layers2.append(nn.Linear(self.latent_size + self.num_classes, self.hidden_dim))
        layers2.append(nn.ReLU())
        layers2.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers2.append(nn.ReLU())
        layers2.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers2.append(nn.ReLU())
        layers2.append(nn.Linear(self.hidden_dim, self.input_size))
        layers2.append(nn.Sigmoid())
        H = int(np.sqrt(self.input_size))
        layers2.append(nn.Unflatten(1, (1, H, H)))
        self.decoder = nn.Sequential(*layers2)
        # ~~END DELETE~~

        #######################################################
        ###               END OF YOUR CODE                  ###
        #######################################################

    def recognition_model(self, x, c):
        """
        Computes the parameters of the posterior distribution q(z | x, c) using the
        recognition network defined in the constructor.

        Inputs:
        - x: PyTorch tensor of shape (batch_size, input_size) for the input data
        - c: PyTorch tensor of shape (batch_size, num_classes) for the input data class

        Returns:
        - mu: PyTorch tensor of shape (batch_size, latent_size) for the posterior mu
        - logvar: PyTorch tensor of shape (batch_size, latent_size) for the posterior
          variance in log space
        """
        #######################################################
        ###              START OF YOUR CODE                 ###
        #######################################################
        # ~~START DELETE~~

        x_flat = torch.flatten(x, start_dim=1)
        x_concat = torch.cat((x_flat, c), 1)
        encoding = self.encoder(x_concat)

        encoding_c = torch.cat((encoding, c), 1)
        mu = self.mu_layer(encoding_c)
        logvar = self.logvar_layer(encoding_c)

        # ~~END DELETE~~
        #######################################################
        ###               END OF YOUR CODE                  ###
        #######################################################

        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std) + mu

    def generation_model(self, z, c): # P(x|z, c)
        """
        Computes the generation output from the generative distribution p(x | z, c)
        using the generation network defined in the constructor

        Inputs:
        - z: PyTorch tensor of shape (batch_size, latent_size) for the latent vector
        - c: PyTorch tensor of shape (batch_size, num_classes) for the input data class

        Returns:
        - x_hat: PyTorch tensor of shape (batch_size, input_size) for the generated data
        """
        #######################################################
        ###              START OF YOUR CODE                 ###
        #######################################################
        # ~~START DELETE~~
        z_c = torch.cat((z, c), 1)
        x_hat = self.decoder(z_c)

        # ~~END DELETE~~
        #######################################################
        ###               END OF YOUR CODE                  ###
        #######################################################
        return x_hat

    def forward(self, x, c):
        """
        Performs the inference and generation steps of the CVAE model using
        the recognition_model, reparametrization trick, and generation_model

        Inputs:
        - x: PyTorch tensor of shape (batch_size, input_size) for the input data
        - c: PyTorch tensor of shape (batch_size, num_classes) for the input data class

        Returns:
        - x_hat: PyTorch tensor of shape (batch_size, input_size) for the generated data
        - mu: PyTorch tensor of shape (batch_size, latent_size) for the posterior mu
        - logvar: PyTorch tensor of shape (batch_size, latent_size)
                  for the posterior logvar
        """
        #######################################################
        ###              START OF YOUR CODE                 ###
        #######################################################
        # ~~START DELETE~~
        mu, logvar = self.recognition_model(x, c)

        z = self.reparametrize(mu, logvar)
        x_hat = self.generation_model(z, c)

        # ~~END DELETE~~
        #######################################################
        ###               END OF YOUR CODE                  ###
        #######################################################
        return x_hat, mu, logvar


def to_var(x, use_cuda):
    x = Variable(x)
    if use_cuda:
        x = x.cuda()
    return x


def one_hot(labels, class_size, use_cuda):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return to_var(targets, use_cuda)


def train(epoch, model, train_loader, optimizer, num_classes, use_cuda):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = to_var(data, use_cuda).view(data.shape[0], -1)
        labels = one_hot(labels, num_classes, use_cuda)
        recon_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data))

def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lowerbound averaged over the minibatch for conditional vae
    Note: We compute -lowerbound because we optimize the network by minimizing a loss

    Inputs:
    - x_hat: PyTorch tensor of shape (batch_size, input_size) for the generated data
    - x: PyTorch tensor of shape (batch_size, input_size) for the real data
    - mu: PyTorch tensor of shape (batch_size, latent_size) for the posterior mu
    - logvar: PyTorch tensor of shape (batch_size, latent_size) for the posterior logvar

    Returns:
    - loss: PyTorch tensor containing the (scalar) loss for the negative lowerbound.
    """
    #######################################################
    ###              START OF YOUR CODE                 ###
    #######################################################
    # ~~START DELETE~~
    N, Z = mu.shape
    x_hat = x_hat.view(N, -1)

    ce_loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kl_loss = 1 + logvar - mu ** 2 - torch.exp(logvar)
    kl_loss = 0.5 * torch.sum(kl_loss)

    # Why is it 2 * ce_loss, and divided by 2 not 1
    # TODO(Jongwook): Verify this quesiton
    loss = (ce_loss - kl_loss) / N

    # ~~END DELETE~~
    #######################################################
    ###               END OF YOUR CODE                  ###
    #######################################################

    return loss


# See cvae.ipynb as well; you can either run the code through juypter notebook
# or directly execute this python code as a script.
use_cuda = False

batch_size = 32
input_size = 28 * 28
hidden_units = 400
latent_size = 20  # z dim
num_classes = 10
num_epochs = 10


def main():
    # Load MNIST dataset
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset = datasets.MNIST(
        './data', train=True, download=True,
        transform=transforms.ToTensor())
    train_dataset = torch.utils.data.Subset(dataset, indices=range(10000))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    model = CVAE(
        input_size=input_size,
        latent_size=latent_size,
        num_classes=num_classes,
        hidden_units=hidden_units,
    )

    if use_cuda:
        model.cuda()

    # Note: You will get an ValueError here if you haven't implemented anything
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    start = time.time()
    for epoch in range(1, num_epochs+1):
        train(epoch, model, train_loader, optimizer, num_classes, use_cuda)
    print('training time = %f' % (time.time() - start)) # should take less than 5 minutes

    # Generate images with condition labels
    c = torch.eye(num_classes, num_classes) # [one hot labels for 0-9]
    c = to_var(c, use_cuda)
    z = to_var(torch.randn(num_classes, latent_size), use_cuda)
    samples = model.generation_model(z, c).data.cpu().numpy()

    fig = plt.figure(figsize=(10, 1))
    gs = gridspec.GridSpec(1, 10)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])  # type: ignore
        ax.set_yticklabels([])  # type: ignore
        ax.set_aspect('equal')  # type: ignore
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    plt.show()


if __name__ == '__main__':
    main()
