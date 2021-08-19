import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from smt.sampling_methods import LHS
from torchdiffeq._impl.fixed_grid import RK4
from torchdyn.numerics import odeint
from tqdm import tqdm


def grid_init_samples(domain, n_trajectories: int) -> np.ndarray:
    """
    :param domain:
        theta min / max
        omega min / max
    :param n_trajectories:
        number of initial value pairs
    """
    x = np.linspace(domain[0][0], domain[0][1], n_trajectories)
    y = np.linspace(domain[1][0], domain[1][1], n_trajectories)

    xx, yy = np.meshgrid(x, y)
    return np.concatenate((xx.flatten()[..., np.newaxis], yy.flatten()[..., np.newaxis]), axis=1)


def random_init_samples(domain, n_trajectories: int) -> np.ndarray:
    """
    :param domain:
        theta min / max
        omega min / max
    :param n_trajectories:
        number of initial value pairs
    """
    values = LHS(xlimits=np.array(domain))
    return values(n_trajectories)


def simulate_ode(f, solver, y0s: torch.Tensor, number_of_steps: int, step_size: float) -> torch.Tensor:
    time_points = torch.arange(
        0., step_size * (number_of_steps + 1), step_size)
    t_eval, ys = odeint(f, y0s, time_points, solver=solver)
    return ys


def simulate_direct(g, y0s: torch.Tensor, number_of_steps: int) -> torch.Tensor:
    ys = [y0s]
    for _ in range(number_of_steps):
        ys.append(g(ys[-1]))
    return torch.stack(ys, dim=1)


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
        self.opt = torch.optim.AdamW(self.net.parameters())

        self.type = self.__class__.__name__
        self.solver = 'rk4'

    def train(self, y0s: torch.Tensor, y: torch.Tensor, number_of_steps: int, step_size: float):
        epochs = 1000
        progress = tqdm(range(epochs), 'Training')
        mses = []
        for _ in progress:
            y_pred = simulate_ode(lambda t, y: self.net(y), self.solver, y0s, number_of_steps, step_size)

            loss = F.mse_loss(y_pred, y)
            mses.append(loss.detach().numpy())
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            progress.set_description(f'loss: {loss.item()}')
        return mses

    def predict(self, y0s, number_of_steps: int, step_size: float):
        y_pred = simulate_ode(lambda t, y: self.net(y), self.solver, y0s, number_of_steps, step_size)
        return y_pred

    def evaluate(self, y):
        return self.net(y)


class HybridModel:

    def __init__(self):
        input_size = 2
        size_of_hidden_layers = 32
        output_size = 1
        self.l = torch.tensor(1., requires_grad=True)
        self.g = torch.tensor(1.)

        self.net = nn.Sequential(
            nn.Linear(input_size, output_size),
        )
        #param = list(self.net.parameters()) + [self.l]
        param = self.net.parameters()
        #self.opt = torch.optim.AdamW([self.l], lr=0.01)
        self.opt = torch.optim.AdamW(param)

        self.type = self.__class__.__name__
        self.solver = 'rk4'

        def f_fric_nn(t, y):
            theta = y[..., 0]
            omega = y[..., 1]
            d_theta = omega
            d_omega = - self.g / self.l * torch.sin(theta) - self.net(y).squeeze()
            return torch.stack([d_theta, d_omega]).T

        self.func = f_fric_nn

    def train(self, y0s: torch.Tensor, y: torch.Tensor, number_of_steps: int, step_size: float):
        epochs = 1000
        progress = tqdm(range(epochs), 'Training for friction')
        mses = []
        ls = []
        for _ in progress:
            y_pred = simulate_ode(self.func, self.solver, y0s, number_of_steps, step_size)
            ls.append(self.l.item())

            loss = F.mse_loss(y_pred, y)
            mses.append(loss.detach().numpy())
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            progress.set_description(f'loss: {loss.item()}')
        return mses#, ls

    def predict(self, y0s, number_of_steps: int, step_size: float):
        y_pred = simulate_ode(self.func, self.solver, y0s, number_of_steps, step_size)
        return y_pred

    def evaluate(self, y):
        return self.func(0.0, y)


def train():

    """
    Generate training data

    """
    y0s_domain = [[-1., 1.], [-1., 1.]]
    y0s_init = torch.tensor(random_init_samples(y0s_domain, 100)).float()
    # plt.scatter(y0s_init[:, 0], y0s_init[:, 1], color="green")
    # plt.ylabel("$\omega$", rotation=0)
    # plt.xlabel("$\\theta$")
    # plt.show()

    number_of_steps_train = 1
    step_size = 0.001

    # k = np.linspace(y0s_domain[0][0], y0s_domain[0][1], 10)
    # m = np.linspace(y0s_domain[1][0], y0s_domain[1][1], 10)
    #
    # xx, yy = np.meshgrid(k, m)
    #
    # theta = xx
    # omega = yy
    # d_theta = omega
    # d_omega = - g / l * np.sin(theta) - (0.2 * omega)
    # plt.streamplot(xx, yy, d_theta, d_omega, color="green")
    # plt.ylabel("$\omega$",rotation=0)
    # plt.xlabel("$\\theta$")
    # plt.show()

    def f_fric(t, y):
        g = torch.tensor(1.)
        l = torch.tensor(1.)
        def friction(w):
            return 0.2 * w
        theta = y[..., 0]
        omega = y[..., 1]
        d_theta = omega
        d_omega = - g / l * torch.sin(theta) - friction(omega)
        return torch.stack([d_theta, d_omega], dim=1)

    def f_non_fric(t, y):
        g = torch.tensor(1.)
        l = torch.tensor(1.)
        theta = y[0]
        omega = y[1]
        d_theta = omega
        d_omega = - g / l * torch.sin(theta)
        return torch.stack([d_theta, d_omega], dim=1)

    y_init = simulate_ode(f_fric, 'rk4', y0s_init, number_of_steps_train, step_size)

    """
    Train

    """
    models = []
    mses = {}
    models.append(HybridModel())
    models.append(DataModel())

    for model in models:

        mse = model.train(
            y0s=y0s_init, y=y_init, number_of_steps=number_of_steps_train, step_size=step_size)
        #plt.plot(ls, color="green")
        #plt.show()

        mses[model.type] = mse

        """
        Validation

        """
        number_of_steps_test = 100
        step_size = 0.001

        y0s = torch.tensor(grid_init_samples(y0s_domain, 10)).float()
        y = simulate_ode(f_fric, 'rk4', y0s, number_of_steps_test, step_size)
        # plt.scatter(y0s[:, 0], y0s[:, 1], color="green")
        # plt.ylabel("$\omega$", rotation=0)
        # plt.xlabel("$\\theta$")
        # plt.show()

        y_pred = model.predict(
            y0s, number_of_steps=number_of_steps_test, step_size=step_size)

        loss = F.mse_loss(y_pred, y)
        print(f'Pred loss = {loss} with a {model.type}')

        """
        Heat Map
        """

        number_of_steps_test = 1
        step_size = 0.001

        y0s = torch.tensor(grid_init_samples(y0s_domain, 100)).float()
        y = simulate_ode(f_fric, 'rk4', y0s, number_of_steps_test, step_size)
        y_deriv = f_fric(0.0, y[1])
        # plt.scatter(y0s[:, 0], y0s[:, 1], color="green")
        # plt.ylabel("$\omega$", rotation=0)
        # plt.xlabel("$\\theta$")
        # plt.show()

        y_pred = model.predict(
            y0s, number_of_steps=number_of_steps_test, step_size=step_size)

        y_pred_deriv = model.evaluate(y_pred[1])
        error = []
        for index, derivs in enumerate(y_deriv):
            error.append(F.mse_loss(y_pred_deriv[index], derivs).detach().numpy())
        
        error = np.array(error)
        error = np.reshape(error, (100, 100))

        
        x_grid = np.reshape(y0s[:, 0].detach().numpy(), (100, 100))
        y_grid = np.reshape(y0s[:, 1].detach().numpy(), (100, 100))
        im = plt.pcolormesh(x_grid, y_grid, error, shading="gouraud", rasterized=True)
        plt.scatter(y0s_init[:, 0], y0s_init[:, 1], color="white", label="initial points", marker = "x")
        plt.ylabel("$\omega$", rotation=0)
        plt.xlabel("$\\theta$")
        plt.colorbar(im)
        plt.legend()
        plt.show()

        """
        Plot results

        """
        # plt.plot(y_pred.detach().numpy()[
        #         :, :, 1], y_pred.detach().numpy()[:, :, 0], color='r')
        # plt.plot(y.numpy()[:, :, 1], y.numpy()[:, :, 0], color='b')
        # plt.scatter(y0s[:, 1], y0s[:, 0])
        # plt.ylim(y0s_domain[0])
        # plt.xlim(y0s_domain[1])
        # plt.show()

        # for name, param in model.net.named_parameters():
        #     print(f"{name}: {param}")

    """
    Plot mse

    """
    for model_type, mse in mses.items():
        plt.plot(mse, label=model_type)
    plt.ylabel("MSE")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train()
