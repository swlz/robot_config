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
        epochs = 3000
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
