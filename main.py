import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from smt.sampling_methods import LHS
from torchdiffeq import odeint
from tqdm import tqdm

g = torch.tensor(9.81)
l = torch.tensor(1.)


def grid_init_samples(domain, n_trajectories: int):
    x = np.linspace(domain[0][0], domain[0][1], n_trajectories)
    y = np.linspace(domain[1][0], domain[1][1], n_trajectories)

    xx, yy = np.meshgrid(x, y)
    return np.concatenate((xx.flatten()[..., np.newaxis], yy.flatten()[..., np.newaxis]), axis=1)


def random_init_samples(domain, n_trajectories: int):
    """
    :param domain:
        theta min / max
        omega min / max
    :param n_trajectories:
        number of initial value pairs
    """
    values = LHS(xlimits=np.array(domain))
    return values(n_trajectories)


def simulate_numerically(y0s: torch.Tensor, number_of_steps: int, step_size: float) -> torch.Tensor:
    time_points = torch.arange(0., step_size * (number_of_steps + 1), step_size)
    ys = [(odeint(f, y0, time_points)[1:]) for y0 in y0s]
    return torch.stack(ys).float()


def f(t, y):
    theta = y[0]
    omega = y[1]
    d_theta = omega
    d_omega = - g / l * torch.sin(theta)
    return torch.tensor([d_theta, d_omega])


def simulate(model, y0s: torch.Tensor, number_of_steps: int) -> torch.Tensor:
    ys = [model(y0s)]
    for _ in range(number_of_steps):
        ys.append(model(ys[-1]))
    return torch.swapaxes(torch.stack(ys), 0, 1)


def simulate_integr(model, y0s: torch.Tensor, number_of_steps: int, step_size: float) -> torch.Tensor:
    ys = [y0s + step_size * model(y0s)]
    for _ in range(number_of_steps):  #
        ys.append(ys[-1] + step_size * model(ys[-1]))
    return torch.swapaxes(torch.stack(ys), 0, 1)


if __name__ == '__main__':
    """
    Generate training data
    
    """
    y0s_domain = [[-1., 1.], [-1., 1.]]
    grid_init_samples(y0s_domain, 10)
    y0s = torch.tensor(random_init_samples(y0s_domain, 1000))

    """
    Data model
    
    """
    input_size = 2
    size_of_hidden_layers = 32
    output_size = 2

    model = nn.Sequential(
        nn.Linear(input_size, size_of_hidden_layers),
        nn.Linear(size_of_hidden_layers, size_of_hidden_layers),
        nn.Linear(size_of_hidden_layers, output_size),
    )

    number_of_steps_train = 1
    step_size = 0.01
    x = y0s.float()
    y = simulate_numerically(y0s, number_of_steps_train, step_size)

    epochs = 1000
    opt = torch.optim.Adam(model.parameters())
    progress = tqdm(range(epochs), 'Training')
    for _ in progress:
        # y_pred = simulate(model, x, n_training)
        y_pred = simulate_integr(model, x, number_of_steps_train, step_size)

        loss = F.mse_loss(y_pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()

        progress.set_description(f'loss: {loss.item()}')

    """
    Validation
    
    """
    y0s = torch.tensor(grid_init_samples(y0s_domain, 10))
    number_of_steps_test = 5
    step_size = 0.1
    x = y0s.float()
    y = simulate_numerically(y0s, number_of_steps_test, step_size)

    # y_pred = simulate(model, n_test, x)
    y_pred = simulate_integr(model, x, number_of_steps_test, step_size)
    loss = F.mse_loss(y_pred, y)
    print(f'Pred loss = {loss}')

    """
    Plot results
    
    """
    plt.plot(y_pred.detach().numpy()[:, :, 0].T, y_pred.detach().numpy()[:, :, 1].T, color='r')
    plt.plot(y.numpy()[:, :, 0].T, y.numpy()[:, :, 1].T, color='b')
    plt.scatter(y0s[:, 0], y0s[:, 1])
    plt.show()
