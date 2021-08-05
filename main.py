import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from smt.sampling_methods import LHS
from torchdiffeq import odeint
from tqdm import tqdm
import matplotlib.pyplot as plt

g = torch.tensor(9.81)
l = torch.tensor(1.)


def grid_init_samples(domain, n_trajectories):
    x = np.linspace(domain[0][0], domain[0][1], n_trajectories)
    y = np.linspace(domain[1][0], domain[1][1], n_trajectories)

    xx, yy = np.meshgrid(x, y)


def random_init_sample(domain, n_trajectories):
    """
    :param domain:
        theta min / max
        omega min / max
    :param n_trajectories:
        number of initial value pairs
    """
    values = LHS(xlimits=np.array(domain))
    return values(n_trajectories)


def generate_data(y0s: torch.Tensor, step_size, n) -> (torch.Tensor, torch.Tensor):
    """
    :param y0s:
        trajectories * states
    :param step_size:
        step size
    :param n:
        number of steps
    :return:
        tensors y0s, ys:
            y0s: trajectories * states
            ys: trajectories * time steps * states
    """
    time_points = torch.arange(0., step_size * (n + 1), step_size)
    ys = [(odeint(f, y0, time_points)[1:]) for y0 in y0s]
    return y0s.float(), torch.stack(ys).float()


def f(t, y):
    theta = y[0]
    omega = y[1]
    d_theta = omega
    d_omega = - g / l * torch.sin(theta)
    return torch.tensor([d_theta, d_omega])


def simulate(model, t, y0s):
    ys = [model(y0s)]
    for tn in range(t - 1):
        ys.append(model(ys[-1]))
    return torch.swapaxes(torch.stack(ys), 0, 1)


if __name__ == '__main__':

    y0s_domain = [[-1., 1.], [-1., 1.]]
    grid_init_samples(y0s_domain, 10)
    y0s = torch.tensor(random_init_sample(y0s_domain, 1000))

    input_size = 2
    size_of_hidden_layers = 32
    output_size = 2

    model = nn.Sequential(
        nn.Linear(input_size, size_of_hidden_layers),
        nn.Linear(size_of_hidden_layers, size_of_hidden_layers),
        nn.Linear(size_of_hidden_layers, output_size),
    )

    n_training = 1
    step_size = 0.01
    x, y = generate_data(y0s, step_size, n_training)

    epochs = 1000
    opt = torch.optim.Adam(model.parameters())
    progress = tqdm(range(epochs), 'Training')
    for _ in progress:
        y_pred = simulate(model, n_training, x)

        loss = F.mse_loss(y_pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()

        progress.set_description(f'loss: {loss.item()}')

    n_training = 50
    step_size = 0.01
    x, y = generate_data(y0s, step_size, n_training)

    y_pred = simulate(model, n_training, x)
    loss = F.mse_loss(y_pred, y)
    print(f'Pred loss = {loss}')

    plt.scatter(y_pred.detach().numpy()[0, :, 0], y_pred.detach().numpy()[0, :, 1])
    plt.scatter(y.numpy()[0, :, 0], y.numpy()[0, :, 1])
    plt.show()
