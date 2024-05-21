import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from hw5_utils import create_samples, generate_sigmas, plot_score



class ScoreNet(nn.Module):
    def __init__(self, n_layers=8, latent_dim=128):
        super().__init__()
        # TODO: Implement the neural network
        # The network has n_layers of linear layers. 
        # Latent dimensions are specified with latent_dim.
        # Between each two linear layers, we use Softplus as the activation layer.
        self.net = nn.Sequential(
            nn.Linear(in_features=3,out_features=latent_dim),
            nn.Softplus(),
            nn.Linear(in_features=latent_dim,out_features=latent_dim),
            nn.Softplus(),
            nn.Linear(in_features=latent_dim,out_features=latent_dim),
            nn.Softplus(),
            nn.Linear(in_features=latent_dim,out_features=latent_dim),
            nn.Softplus(),
            nn.Linear(in_features=latent_dim,out_features=latent_dim),
            nn.Softplus(),
            nn.Linear(in_features=latent_dim,out_features=latent_dim),
            nn.Softplus(),
            nn.Linear(in_features=latent_dim,out_features=latent_dim),
            nn.Softplus(),
            nn.Linear(in_features=latent_dim,out_features=2),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                # init weight and bias
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)


    def forward(self, x, sigmas):
        """.
        Parameters
        ----------
        x : torch.tensor, N x 2

        sigmas : torch.tensor of shape N x 1 or a float number
        """
        if isinstance(sigmas, float):
            sigmas = torch.tensor(sigmas).reshape(1, 1).repeat(x.shape[0], 1)
        if sigmas.dim() == 0:
            sigmas = sigmas.reshape(1, 1).repeat(x.shape[0], 1)
        sigmas = sigmas.view(-1,1)
        # we use the trick from NCSNv2 to explicitly divide sigma
        return self.net(torch.concatenate([x, sigmas], dim=-1)) / sigmas


def compute_denoising_loss(scorenet, training_data, sigmas):
    """
    Compute the denoising loss.
    Parameters
    ----------
    scorenet : nn.Module
        The neural network for score prediction

    training_data : np.array, N x 2
        The training data

    sigmas : np.array, L
        The list of sigmas

    Return
    ------
    loss averaged over all training data
    """
    B, C = training_data.shape
    # TODO: Implement the denoising loss follow the steps: 
    # For each training sample x: 
    # 1. Randomly sample a sigma from sigmas
    # 2. Perturb the training sample: \tilde(x) = x + sigma * z
    # 3. Get the predicted score
    # 4. Compute the loss: 1/2 * lambda * ||score + ((\tilde(x) - x) / sigma^2)||^2
    # Return the loss averaged over all training samples
    # Note: use batch operations as much as possible to avoid iterations
    
    # Randomly sample a sigma for each data point
    sigma_indices = torch.randint(0, len(sigmas), (B,))
    sampled_sigmas = sigmas[sigma_indices]
    
    # Perturb the training sample
    z = torch.randn_like(training_data)
    tilde_x = training_data + sampled_sigmas[:, None] * z
    # Get the predicted score
    scores = scorenet(tilde_x, sampled_sigmas)

    # Compute the loss
    lambda_val = sampled_sigmas[:, None] ** 2
    tmp = scores + ((tilde_x - training_data) / sampled_sigmas[:, None] ** 2)
    result = torch.sum(tmp ** 2, dim=1, keepdim=True)
    loss = 0.5 * lambda_val * result
    # Return the loss averaged over all training samples
    loss = loss.mean()
    return loss


@torch.no_grad()
def langevin_dynamics_sample(scorenet, n_samples, sigmas, iterations=100, eps=0.00002, return_traj=False):
    """
    Sample with langevin dynamics.
    Parameters
    ----------
    scorenet : nn.Module
        The neural network for score prediction
    n_samples: int
        Number of samples to acquire
    sigmas : np.array, L
        The list of sigmas
    iterations: int
        The number of iterations for each sigma (T in Alg. 2)
    eps: float
        The parameter to control step size
    return_traj: bool, default is False
        If True, return all intermediate samples
        If False, only return the last step

    Return
    ------
    torch.Tensor in the shape of n_samples x 2 if return_traj=False
    in the shape of n_samples x (L*T) x 2 if return_traj=True
    """

    # TODO: Implement the Langevin dynamics following the steps:
    # 1. Initialize x_0 ~ N(0, I)
    # 2. Iterate through sigmas, for each sigma:
    # 3.    Compute alpha = eps * sigma^2 / sigmaL^2
    # 4.    Iterate through T steps:
    # 5.        x_t = x_{t-1} + alpha * scorenet(x_{t-1}, sigma) + sqrt(2 * alpha) * z
    # 6.    x_0 = x_T
    # 7. Return the last x_T if return_traj=False, or return all x_t
    with torch.no_grad():
        sigmaL2 = sigmas[-1]**2
        # Initialize x_0 ~ N(0, I)
        x_0 = torch.randn((n_samples,2))
        trajectory = []
        #Iterate through sigmas, for each sigma:
        for sigma in sigmas:
            alpha = eps * (sigma**2) / sigmaL2
            x_t = x_0
            for _ in range(iterations):
                z = torch.randn_like(x_t)
                score = scorenet(x_t, sigma)
                x_t = x_t + alpha * score + torch.sqrt(2 * alpha) * z
                if return_traj:
                    trajectory.append(x_t)
            # Set for the next sigma
            x_0 = x_t  

        if return_traj:
            # Concatenate trajectory along the first dimension
            trajectory_expanded = [t.unsqueeze(1) for t in trajectory]
            ret = torch.cat(trajectory_expanded, dim=1)
            print('ret shape is',ret.size())
            return ret
        else:
            return x_0


def main():
    # training related hyperparams
    lr = 0.01
    n_iters = 50000
    log_freq = 1000

    # sampling related hyperparams
    n_samples = 1000
    sample_iters = 100
    sample_lr = 0.00002

    # create the training set
    training_data = torch.tensor(create_samples()).float()

    # visualize the training data
    plt.figure(figsize=(20, 5))
    plt.scatter(training_data[:, 0], training_data[:, 1])
    plt.axis('scaled')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-0.6, 0.6)
    plt.show()


    # create ScoreNet and optimizer
    scorenet = ScoreNet()
    scorenet.train()
    optimizer = optim.Adam(scorenet.parameters(), lr=lr)

    # generate sigmas in descending order: sigma1 > sigma2 > ... > sigmaL
    sigmas = torch.tensor(generate_sigmas(0.3, 0.01, 10)).float()

    avg_loss = 0.
    for i_iter in range(n_iters):
        optimizer.zero_grad()
        loss = compute_denoising_loss(scorenet, training_data, sigmas)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        if i_iter % log_freq == log_freq - 1:
            avg_loss /= log_freq
            print(f'iter {i_iter}: loss = {avg_loss:.3f}')
            avg_loss = 0.

    torch.save(scorenet.state_dict(), 'model.ckpt')
    
    # Q5(a). visualize score function
    scorenet.eval()
    # print('shape is',training_data.size())
    plot_score(scorenet, training_data)

    # Q5(b). sample with langevin dynamics
    samples = langevin_dynamics_sample(scorenet, n_samples, sigmas, sample_iters, sample_lr, return_traj=True).numpy()
    # print(samples)
    # plot the samples
    for step in range(0, sample_iters * len(sigmas), 200):
        plt.figure(figsize=(20, 5))
        plt.scatter(samples[:, step, 0], samples[:, step, 1], color='red')
        plt.axis('scaled')
        plt.xlim(-2.5, 2.5)
        plt.ylim(-0.6, 0.6)
        plt.title(f'Samples at step={step}')
        plt.savefig(f'step_{step}.png', bbox_inches='tight')
        plt.show()
        

    plt.figure(figsize=(20, 5))
    plt.scatter(samples[:, -1, 0], samples[:, -1, 1], color='red')
    plt.axis('scaled')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-0.6, 0.6)
    plt.title('All samples')
    plt.savefig(f'all.png', bbox_inches='tight')
    plt.show()
    

    # Q5(c). visualize the trajectory
    traj = langevin_dynamics_sample(scorenet, 2, sigmas, sample_iters, sample_lr, return_traj=True).numpy()
    plt.figure(figsize=(20, 5))
    plt.plot(traj[0, :, 0], traj[0, :, 1], color='blue')
    plt.plot(traj[1, :, 0], traj[1, :, 1], color='green')

    plt.scatter(samples[:, -1, 0], samples[:, -1, 1], color='red')
    plt.axis('scaled')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-0.6, 0.6)
    plt.title('Trajectories')
    plt.savefig(f'trajectories.png', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    main()