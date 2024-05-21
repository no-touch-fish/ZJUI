import hw4
import hw4_utils
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def main():

    # initialize parameters
    lr = 0.01
    latent_dim = 6
    lam = 5e-5
    loss_fn = nn.MSELoss()
    n_iter = 8000

    # initialize model
    vae = hw4.VAE(lam=lam, lrate=lr, latent_dim=latent_dim, loss_fn=loss_fn)

    # generate data
    X = hw4_utils.generate_data()

    # fit the model to the data
    loss_rec, loss_kl, loss_total, Xhat, gen_samples = hw4.fit(vae, X, n_iter)
    # print('loss_rec is',loss_rec,'loss_kl is',loss_kl,'loss is',loss_total)
    torch.save(vae.cpu().state_dict(),"vae.pb")
    
    # change torch to numpy first for ploting
    loss_total = [loss.item() for loss in loss_total]
    loss_kl = [loss.item() for loss in loss_kl]
    loss_rec = [loss.item() for loss in loss_rec]
    # loss_total = [loss.detach().cpu().numpy() for loss in loss_total] 
    X = X.detach().cpu().numpy()
    Xhat = Xhat.detach().cpu().numpy()
    gen_samples = gen_samples.detach().cpu().numpy()

    # plot the loss vs iteration count
    plt.plot(loss_total)
    plt.title('Loss vs. Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    # test
    plt.plot(loss_kl)
    plt.title('Loss_kl vs. Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    plt.plot(loss_rec)
    plt.title('Loss_rec vs. Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()


    # plot the data points and approximations
    plt.scatter(X[:, 0], X[:, 1], color='blue', label='data points')
    plt.scatter(Xhat[:, 0], Xhat[:, 1], color='red', label='approximations')
    plt.legend()
    plt.title('Plot of the data points and approximations')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.show()
    # plot the data points and approximations and generated points
    plt.scatter(X[:, 0], X[:, 1], color='blue', label='data points')
    plt.scatter(Xhat[:, 0], Xhat[:, 1], color='red', label='approximations')
    plt.scatter(gen_samples[:, 0], gen_samples[:, 1], color='yellow', label='generated points')
    plt.legend()
    plt.title('Plot of the data points and approximations and generated points')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.show()

if __name__ == "__main__":
    main()
