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


def simulate_ode(f, y0s: torch.Tensor, number_of_steps: int, step_size: float) -> torch.Tensor:
    time_points = torch.arange(0., step_size * (number_of_steps + 1), step_size)
    ys = [(odeint(f, y0, time_points)) for y0 in y0s]
    return torch.stack(ys).float()


def simulate_direct(g, y0s: torch.Tensor, number_of_steps: int) -> torch.Tensor:
    ys = [y0s]
    for _ in range(number_of_steps):
        ys.append(g(ys[-1]))
    return torch.swapaxes(torch.stack(ys), 0, 1)


def simulate_euler(f, y0s: torch.Tensor, number_of_steps: int, step_size: float) -> torch.Tensor:
    ys = [y0s]
    for _ in range(number_of_steps):
        ys.append(ys[-1] + step_size * f(ys[-1]))
    return torch.swapaxes(torch.stack(ys), 0, 1)


class DataModel:

    def __init__(self):
        input_size = 2
        size_of_hidden_layers = 32
        output_size = 2

        self.net = nn.Sequential(
            nn.Linear(input_size, size_of_hidden_layers),
            nn.Linear(size_of_hidden_layers, size_of_hidden_layers),
            nn.Linear(size_of_hidden_layers, output_size),
        )
        self.opt = torch.optim.Adam(self.net.parameters())

        self.type = "Data Model"

    def train(self, x: torch.Tensor, y: torch.Tensor, number_of_steps: int, step_size: float):
        epochs = 2000
        progress = tqdm(range(epochs), 'Training')
        mses = []
        for _ in progress:
            y_pred = simulate_euler(self.net, x, number_of_steps, step_size)

            loss = F.mse_loss(y_pred, y)
            mses.append(loss.detach().numpy())
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            progress.set_description(f'loss: {loss.item()}')
        return mses

    def predict(self, y0s, number_of_steps: int, step_size: float):
        y_pred = simulate_euler(self.net, y0s, number_of_steps, step_size)
        return y_pred


class HybridModel:

    def __init__(self):
        input_size = 1
        size_of_hidden_layers = 32
        output_size = 1

        self.net = nn.Sequential(
            nn.Linear(input_size, output_size),
            #nn.Linear(input_size, size_of_hidden_layers),
            # nn.Softplus(),
            # nn.Linear(size_of_hidden_layers, size_of_hidden_layers),
            # nn.Softplus(),
            # nn.Linear(size_of_hidden_layers, output_size),
        )
        self.opt = torch.optim.Adam(self.net.parameters())

        self.type = "Hybrid Model"

        def f_fric_nn(y):
            theta = y[:, 0]
            omega = y[:, 1]
            d_theta = omega
            d_omega = - g / l * torch.sin(theta) - self.net(omega.unsqueeze(dim=1)).squeeze()
            return torch.stack([d_theta, d_omega]).T

        self.func = f_fric_nn

    def train(self, x: torch.Tensor, y: torch.Tensor, number_of_steps: int, step_size: float):
        epochs = 2000
        progress = tqdm(range(epochs), 'Training for friction')
        mses = []
        for _ in progress:
            y_pred = simulate_euler(self.func, x, number_of_steps, step_size)

            loss = F.mse_loss(y_pred, y)
            mses.append(loss.detach().numpy())
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            progress.set_description(f'loss: {loss.item()}')
        return mses

    def predict(self, y0s, number_of_steps: int, step_size: float):
        y_pred = simulate_euler(self.func, y0s, number_of_steps, step_size)
        return y_pred


if __name__ == '__main__':
    """
    Generate training data
    
    """
    y0s_domain = [[-1., 1.], [-1., 1.]]
    y0s_init = torch.tensor(random_init_samples(y0s_domain, 1000)).float()

    number_of_steps_train = 1
    step_size = 0.001


    def f_fric(t, y):
        def friction(w):
            return 0.2 * w
        theta = y[0]
        omega = y[1]
        d_theta = omega
        d_omega = - g / l * torch.sin(theta) - friction(omega)
        return torch.tensor([d_theta, d_omega])

    def f_non_fric(t, y):
        theta = y[0]
        omega = y[1]
        d_theta = omega
        d_omega = - g / l * torch.sin(theta)
        return torch.tensor([d_theta, d_omega])

    y_init = simulate_ode(f_fric, y0s_init, number_of_steps_train, step_size)

    """
    Train
    
    """
    models = []
    mses = []
    models.append(HybridModel())
    models.append(DataModel())

    for model in models:

        mse = model.train(x=y0s_init, y=y_init, number_of_steps=number_of_steps_train, step_size=step_size)
        mses.append(mse)

        """
        Validation
        
        """
        number_of_steps_test = 100
        step_size = 0.001

        y0s = torch.tensor(grid_init_samples(y0s_domain, 10)).float()
        y = simulate_ode(f_fric, y0s, number_of_steps_test, step_size)

        y_pred = model.predict(y0s, number_of_steps=number_of_steps_test, step_size=step_size)

        loss = F.mse_loss(y_pred, y)
        print(f'Pred loss = {loss} with a {model.type}')

        """
        Plot results
        
        """
        plt.plot(y_pred.detach().numpy()[:, :, 1].T, y_pred.detach().numpy()[:, :, 0].T, color='r')
        plt.plot(y.numpy()[:, :, 1].T, y.numpy()[:, :, 0].T, color='b')
        plt.scatter(y0s[:, 1], y0s[:, 0])
        plt.ylim(y0s_domain[0])
        plt.xlim(y0s_domain[1])
        plt.show()

        print([param for param in model.net.parameters()])

    """
    Plot mse

    """

    plt.plot(mses[0], color='r')
    plt.plot(mses[1], color='b')
    plt.ylabel("MSE")
    plt.xlabel("Epochs")
    plt.show()

