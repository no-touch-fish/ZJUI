import hw4_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

class VAE(torch.nn.Module):
    def __init__(self, lam,lrate,latent_dim,loss_fn):
        """
        Initialize the layers of your neural network

        @param lam: Hyperparameter to scale KL-divergence penalty
        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param latent_dim: The dimension of the latent space

        The network should have the following architecture (in terms of hidden units):
        Encoder Network:
        2 -> 50 -> ReLU -> 50 -> ReLU -> 50-> ReLU -> (6,6) (mu_layer,logstd2_layer)

        Decoder Network:
        6 -> 50 -> ReLU -> 50 -> ReLU -> 2 -> Sigmoid

        See set_parameters() function for the exact shapes for each weight
        """
        super(VAE, self).__init__()
        # code start
        self.encoder_linear_1 = nn.Linear(in_features=2, out_features=50)
        self.encoder_linear_2 = nn.Linear(in_features=50, out_features=50)
        self.encoder_linear_3 = nn.Linear(in_features=50, out_features=50)
        self.encoder_linear_mu = nn.Linear(in_features=50, out_features=latent_dim)
        self.encoder_linear_std = nn.Linear(in_features=50, out_features=latent_dim)

        self.decoder_linear_1 = nn.Linear(in_features=latent_dim, out_features=50)
        self.decoder_linear_2 = nn.Linear(in_features=50, out_features=50)
        self.decoder_linear_3 = nn.Linear(in_features=50, out_features=2)

        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        # code end
        self.lrate = lrate
        self.lam = lam
        self.loss_fn = loss_fn
        self.latent_dim = latent_dim
        self.optimizer = optim.Adam(self.parameters(), lr=self.lrate)

    def set_parameters(self, We1,be1, We2, be2, We3, be3, Wmu, bmu, Wstd, bstd, Wd1, bd1, Wd2, bd2, Wd3, bd3):
        """ Set the parameters of your network

        # Encoder weights:
        @param We1: an (50,2) torch tensor
        @param be1: an (50,) torch tensor
        @param We2: an (50,50) torch tensor
        @param be2: an (50,) torch tensor
        @param We3: an (50,50) torch tensor
        @param be3: an (50,) torch tensor
        @param Wmu: an (6,50) torch tensor
        @param bmu: an (6,) torch tensor
        @param Wstd: an (6,50) torch tensor
        @param bstd: an (6,) torch tensor

        # Decoder weights:
        @param Wd1: an (50,6) torch tensor
        @param bd1: an (50,) torch tensor
        @param Wd2: an (50,50) torch tensor
        @param bd2: an (50,) torch tensor
        @param Wd3: an (2,50) torch tensor
        @param bd3: an (2,) torch tensor

        """
        # code start
        with torch.no_grad():
            self.encoder_linear_1.weight = nn.Parameter(We1)
            self.encoder_linear_1.bias = nn.Parameter(be1)
            self.encoder_linear_2.weight = nn.Parameter(We2)
            self.encoder_linear_2.bias = nn.Parameter(be2)
            self.encoder_linear_3.weight = nn.Parameter(We3)
            self.encoder_linear_3.bias = nn.Parameter(be3)
            self.encoder_linear_mu.weight = nn.Parameter(Wmu)
            self.encoder_linear_mu.bias = nn.Parameter(bmu)
            self.encoder_linear_std.weight = nn.Parameter(Wstd)
            self.encoder_linear_std.bias = nn.Parameter(bstd)

            self.decoder_linear_1.weight = nn.Parameter(Wd1)
            self.decoder_linear_1.bias = nn.Parameter(bd1)
            self.decoder_linear_2.weight = nn.Parameter(Wd2)
            self.decoder_linear_2.bias = nn.Parameter(bd2)
            self.decoder_linear_3.weight = nn.Parameter(Wd3)
            self.decoder_linear_3.bias = nn.Parameter(bd3)
        # code end
    def forward(self, x):
        """ A forward pass of your autoencoder

        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return y: an (N, 50) torch tensor of output from the encoder network
        @return mean: an (N,latent_dim) torch tensor of output mu layer
        @return stddev_p: an (N,latent_dim) torch tensor of output stddev layer
        @return z: an (N,latent_dim) torch tensor sampled from N(mean,exp(stddev_p/2)
        @return xhat: an (N,D) torch tensor of outputs from f_dec(z)
        """
        # code start
        y = self.ReLU(self.encoder_linear_1(x))
        y = self.ReLU(self.encoder_linear_2(y))
        y = self.ReLU(self.encoder_linear_3(y))
        mean = self.encoder_linear_mu(y)
        stddev_p = self.encoder_linear_std(y)
        stddev = torch.exp(stddev_p / 2)
        epsilon = torch.randn(mean.shape)
        z = mean + epsilon * stddev
        xhat = self.ReLU(self.decoder_linear_1(z))
        xhat = self.ReLU(self.decoder_linear_2(xhat))
        xhat = self.Sigmoid(self.decoder_linear_3(xhat))
        # code end
        return y,mean,stddev_p,z,xhat

    def step(self, x):
        """
        Performs one gradient step through a batch of data x
        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return L_rec: float containing the reconstruction loss at this time step
        @return L_kl: kl divergence penalty at this time step
        @return L: total loss at this time step
        """
        # code start
        N,_ = x.size()
        y,mean,stddev_p,z,xhat = self.forward(x)
        var = torch.exp(stddev_p) 
        L_rec = self.loss_fn(xhat,x)
        # kl divergence
        kl_div = 0.5 * (var + mean.pow(2) - 1 - stddev_p)
        kl_div_sum = kl_div.sum(dim=1)
        L_kl = kl_div_sum.mean()

        L = L_rec + (self.lam * L_kl)
        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()
        # code end
        return L_rec,L_kl,L
    
    def generate(self,z):
        gen_samples = self.ReLU(self.decoder_linear_1(z))
        gen_samples = self.ReLU(self.decoder_linear_2(gen_samples))
        gen_samples = self.Sigmoid(self.decoder_linear_3(gen_samples))
        return gen_samples


def fit(net,X,n_iter):
    """ Fit a VAE.  Use the full batch size.
    @param net: the VAE
    @param X: an (N, D) torch tensor
    @param n_iter: int, the number of iterations of training

    # return all of these from left to right:

    @return losses_rec: Array of reconstruction losses at the beginning and after each iteration. Ensure len(losses_rec) == n_iter
    @return losses_kl: Array of KL loss penalties at the beginning and after each iteration. Ensure len(losses_kl) == n_iter
    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return Xhat: an (N,D) NumPy array of approximations to X
    @return gen_samples: an (N,D) NumPy array of N samples generated by the VAE
    """
    # code start
    N,D = X.size()
    losses_rec = []
    losses_kl = []
    losses = []
    for i in range(n_iter):
        L_rec,L_kl,L = net.step(X)
        losses_rec.append(L_rec)
        losses_kl.append(L_kl)
        losses.append(L)
    y,mean,stddev_p,z,xhat = net.forward(X)
    z = torch.randn_like(z)
    gen_samples = net.generate(z)

    # code end
    return losses_rec,losses_kl,losses,xhat,gen_samples
