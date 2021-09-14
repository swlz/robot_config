import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from smt.sampling_methods import LHS
from torchdiffeq._impl.fixed_grid import RK4
from torchdyn.numerics import odeint
from tqdm import tqdm
import HybridModel as HM
import DataModel as DM

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


def f_fric(t, y):
    g = torch.tensor(1.)
    l = torch.tensor(1.)
    def friction_coef(w):
        return 0.2 * w
    def friction_sin(w):
        return torch.sin(w)
    def friction_quadratic(w):
        return w ** 2
    def friction_cubic(w):
        return w ** 3
    def friction_gauss(w):
        return torch.normal(1, 1, size=w.size())
    theta = y[..., 0]
    omega = y[..., 1]
    d_theta = omega
    d_omega = - g / l * torch.sin(theta) - friction_gauss(omega)
    return torch.stack([d_theta, d_omega], dim=1)

def f_non_fric(t, y):
    g = torch.tensor(1.)
    l = torch.tensor(1.)
    theta = y[0]
    omega = y[1]
    d_theta = omega
    d_omega = - g / l * torch.sin(theta)
    return torch.stack([d_theta, d_omega], dim=1)


def plot_y0s(y0s):
    plt.scatter(y0s[:, 0], y0s[:, 1], color="green")
    plt.ylabel("$\omega$", rotation=0)
    plt.xlabel("$\\theta$")
    plt.show()

def plot_heatmap(model, y0s_domain, y0s_init):
    number_of_steps_test = 1
    step_size = 0.001

    y0s = torch.tensor(grid_init_samples(y0s_domain, 100)).float()
    y = simulate_ode(f_fric, 'rk4', y0s, number_of_steps_test, step_size)
    y_deriv = f_fric(0.0, y[1])

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

def plot_prediction(y0s, y, y_pred, y0s_domain):
    plt.plot(y_pred.detach().numpy()[
            :, :, 1], y_pred.detach().numpy()[:, :, 0], color='r')
    plt.plot(y.numpy()[:, :, 1], y.numpy()[:, :, 0], color='b')
    plt.scatter(y0s[:, 1], y0s[:, 0])
    plt.ylim(y0s_domain[0])
    plt.xlim(y0s_domain[1])
    plt.show()

def plot_mse(mses):
    for model_type, mse in mses.items():
        plt.plot(mse, label=model_type)
        plt.ylabel("MSE")
        plt.xlabel("Epochs")
        plt.legend()
    plt.show()

def plot_grid(y0s_domain, g, l):
    k = np.linspace(y0s_domain[0][0], y0s_domain[0][1], 10)
    m = np.linspace(y0s_domain[1][0], y0s_domain[1][1], 10)
    
    xx, yy = np.meshgrid(k, m)
    
    theta = xx
    omega = yy
    d_theta = omega
    d_omega = - g / l * np.sin(theta) - (0.2 * omega)
    plt.streamplot(xx, yy, d_theta, d_omega, color="green")
    plt.ylabel("$\omega$",rotation=0)
    plt.xlabel("$\\theta$")
    plt.show()


def train():

    """
    Generate training data

    """
    y0s_domain = [[-1., 1.], [-1., 1.]]
    y0s_init = torch.tensor(random_init_samples(y0s_domain, 100)).float()

    number_of_steps_train = 1
    step_size = 0.001

    y_init = simulate_ode(f_fric, 'rk4', y0s_init, number_of_steps_train, step_size)

    """
    Train

    """
    models = []
    mses = {}
    models.append(HM.HybridModel())
    models.append(DM.DataModel())

    for model in models:

        mse = model.train(
            y0s=y0s_init, y=y_init, number_of_steps=number_of_steps_train, step_size=step_size)
        mses[model.type] = mse

        """
        Validation

        """
        number_of_steps_test = 100
        step_size = 0.001

        y0s = torch.tensor(grid_init_samples(y0s_domain, 10)).float()
        y = simulate_ode(f_fric, 'rk4', y0s, number_of_steps_test, step_size)
        y_pred = model.predict(
            y0s, number_of_steps=number_of_steps_test, step_size=step_size)

        loss = F.mse_loss(y_pred, y)
        print(f'Pred loss = {loss} with a {model.type}')

        """
        Heat Map
        """

        plot_heatmap(model, y0s_domain, y0s_init)


        """
        Plot results

        """
        plot_prediction(y0s, y, y_pred, y0s_domain)

        # for name, param in model.net.named_parameters():
        #     print(f"{name}: {param}")

    """
    Plot mse

    """
    plot_mse(mses)

if __name__ == '__main__':
    train()
