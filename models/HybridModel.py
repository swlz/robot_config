import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdyn.numerics import odeint
from tqdm import tqdm



def simulate_ode(f, solver, y0s: torch.Tensor, number_of_steps: int, step_size: float) -> torch.Tensor:
    time_points = torch.arange(
        0., step_size * (number_of_steps + 1), step_size)
    t_eval, ys = odeint(f, y0s, time_points, solver=solver)
    return ys

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
        epochs = 3000
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
