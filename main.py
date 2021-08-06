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

    def train(self, x: torch.Tensor, y: torch.Tensor, number_of_steps: int, step_size: float) -> torch.nn:
        input_size = 2
        size_of_hidden_layers = 32
        output_size = 2

        net = nn.Sequential(
            nn.Linear(input_size, size_of_hidden_layers),
            nn.Linear(size_of_hidden_layers, size_of_hidden_layers),
            nn.Linear(size_of_hidden_layers, output_size),
        )

        epochs = 1000
        opt = torch.optim.Adam(net.parameters())
        progress = tqdm(range(epochs), 'Training')
        for _ in progress:
            y_pred = simulate_euler(net, x, number_of_steps, step_size)

            loss = F.mse_loss(y_pred, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

            progress.set_description(f'loss: {loss.item()}')
        return net

    def predict(self, net: torch.nn, number_of_steps: int, step_size: float):
        y0s = torch.tensor(grid_init_samples(y0s_domain, 10))
        x = y0s.float()
        y = simulate_ode(f_fric, y0s, number_of_steps_test, step_size)

        y_pred = simulate_euler(net, x, number_of_steps, step_size)
        # y_pred = simulate_numerically(model, x, number_of_steps_test, step_size)
        loss = F.mse_loss(y_pred, y)
        print(f'Pred loss = {loss}')
        return y_pred, y, x


class HybridModel:

    def train(self, x: torch.Tensor, y: torch.Tensor, number_of_steps: int, step_size: float) -> torch.nn:
        input_size = 2
        size_of_hidden_layers = 32
        output_size = 2

        net = nn.Sequential(
            nn.Linear(input_size, size_of_hidden_layers),
            nn.Linear(size_of_hidden_layers, size_of_hidden_layers),
            nn.Linear(size_of_hidden_layers, output_size),
        )

        epochs = 1000
        opt = torch.optim.Adam(net.parameters())
        progress = tqdm(range(epochs), 'Training')
        for _ in progress:
            y_pred = simulate_euler(net, x, number_of_steps, step_size)

            loss = F.mse_loss(y_pred, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

            progress.set_description(f'loss: {loss.item()}')
        return net

    def friction(self, x: torch.Tensor, y: torch.Tensor, number_of_steps: int, step_size: float):
        input_size = 1
        size_of_hidden_layers = 32
        output_size = 1

        net = nn.Sequential(
            nn.Linear(input_size, size_of_hidden_layers),
            nn.Linear(size_of_hidden_layers, size_of_hidden_layers),
            nn.Linear(size_of_hidden_layers, output_size),
        )

        def f_fric_nn(t, y):
            theta = y[0]
            omega = y[1]
            d_theta = omega
            d_omega = - g / l * torch.sin(theta) - net(omega.unsqueeze(dim=0))
            return torch.tensor([d_theta, d_omega])

        epochs = 10
        opt = torch.optim.Adam(net.parameters())
        progress = tqdm(range(epochs), 'Training for friction')
        for _ in progress:
            y_pred = simulate_ode(f_fric_nn, x, number_of_steps, step_size)

            loss = F.mse_loss(y_pred, y)
            loss.requires_grad = True
            loss.backward()
            opt.step()
            opt.zero_grad()

            progress.set_description(f'loss: {loss.item()}')
        return y_pred

    def predict(self, net: torch.nn, number_of_steps: int, step_size: float):
        y0s = torch.tensor(grid_init_samples(y0s_domain, 10))
        x = y0s.float()
        y = simulate_ode(f_fric, y0s, number_of_steps, step_size)

        #y_pred = simulate_euler(net, x, number_of_steps, step_size)
        y_pred = self.friction(x, y, number_of_steps, step_size)
        loss = F.mse_loss(y_pred, y)
        print(f'Pred loss = {loss}')
        return y_pred, y, x


if __name__ == '__main__':
    """
    Generate training data
    
    """
    y0s_domain = [[-1., 1.], [-2., 2.]]
    grid_init_samples(y0s_domain, 10)
    y0s = torch.tensor(random_init_samples(y0s_domain, 1000))

    number_of_steps_train = 1
    step_size = 0.01


    def f_fric(t, y):
        def friction(w):
            return 0.1 * w
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


    x = y0s.float()
    y = simulate_ode(f_fric, y0s, number_of_steps_train, step_size)

    """
    Train
    
    """
    model = HybridModel()
    network = model.train(x=x, y=y, number_of_steps=number_of_steps_train, step_size=step_size)

    """
    Validation
    
    """
    number_of_steps_test = 50
    step_size = 0.01
    y_pred, y, y0s = model.predict(net=network, number_of_steps=number_of_steps_test, step_size=step_size)

    """
    Plot results
    
    """
    plt.plot(y_pred.detach().numpy()[:, :, 1].T, y_pred.detach().numpy()[:, :, 0].T, color='r')
    plt.plot(y.numpy()[:, :, 1].T, y.numpy()[:, :, 0].T, color='b')
    plt.scatter(y0s[:, 1], y0s[:, 0])
    plt.ylim(y0s_domain[0])
    plt.xlim(y0s_domain[1])
    plt.show()
