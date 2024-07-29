# Provides classes for the rejection steps and for a stack of neural JKO (=CNF)
# or rejection steps.

import torch
import math
import numpy as np
from src.CNF_utils import CNF


device = "cuda" if torch.cuda.is_available() else "cpu"


class FlowModel(torch.nn.Module):
    def __init__(self, dim, batch_size=25000, latent_mean=None, latent_scale=1.0):
        super(FlowModel, self).__init__()
        self.step_stack = []
        self.batch_size = batch_size
        self.dim = dim
        self.sample_stack = []
        self.sample_energy_stack = []
        self.stack_size = batch_size
        if latent_mean is None:
            latent_mean = torch.zeros(dim, dtype=torch.float, device=device)
        self.latent_mean = latent_mean
        self.latent_scale = latent_scale

    def latent_energy(self, x):
        x_energy = 0.5 / self.latent_scale**2 * torch.sum(
            x**2, -1
        ) + 0.5 * self.dim * np.log(2 * math.pi * self.latent_scale**2)
        return x_energy

    def sample_latent(self, N):
        x = self.latent_scale * torch.randn(
            (N, self.dim), device=device, dtype=torch.float
        )
        x_energy = self.latent_energy(x)
        x = x + self.latent_mean[None, :]
        return x, x_energy

    def sample(self, N, stopping_step=None):
        # sampling procedure. In order avoid sampling calls with very few
        # samples during the rejection-resampling steps, we always draw "stack_size" samples
        # and store the unused ones for the next call
        if stopping_step is None:
            stopping_step = len(self.step_stack)
        step_ind = stopping_step - 1
        if step_ind < 0:
            return self.sample_latent(N)
        while self.sample_stack[step_ind].shape[0] < N:
            x, x_energy = self.sample(self.stack_size, stopping_step=step_ind)
            x, x_energy = self(x, x_energy, step_index=step_ind)
            self.sample_stack[step_ind] = torch.cat((self.sample_stack[step_ind], x), 0)
            self.sample_energy_stack[step_ind] = torch.cat(
                (self.sample_energy_stack[step_ind], x_energy), 0
            )
        x = self.sample_stack[step_ind][:N]
        self.sample_stack[step_ind] = self.sample_stack[step_ind][N:]
        x_energy = self.sample_energy_stack[step_ind][:N]
        self.sample_energy_stack[step_ind] = self.sample_energy_stack[step_ind][N:]
        return x, x_energy

    def energy_evaluation(self, x, stopping_step=None):
        # energy (= negative log-density) evaluation of the model until step stopping_step
        x = x.clone()
        if stopping_step is None:
            stopping_step = len(self.step_stack)
        with torch.no_grad():
            x_steps = [x.clone()]
            x_energy_transforms = []
            for i in range(stopping_step - 1, -1, -1):
                if not isinstance(self.step_stack[i], CNF):
                    x_steps.insert(0, x.clone())
                    x_energy_transforms.insert(0, None)
                else:
                    x_e_trafo = torch.zeros_like(x[:, 0])
                    j = 0
                    while j < x.shape[0]:
                        batch = x[j : j + self.batch_size]
                        out = self.step_stack[i](
                            batch,
                            torch.zeros_like(x_e_trafo[j : j + self.batch_size]),
                            reverse=True,
                        )
                        x[j : j + self.batch_size] = out[0]
                        x_e_trafo[j : j + self.batch_size] = out[2].squeeze()
                        j = j + self.batch_size
                    x_steps.insert(0, x.clone().detach())
                    x_energy_transforms.insert(0, x_e_trafo.detach())

            x = x_steps[0]
            x_energy = self.latent_energy(x)
            x_steps = x_steps[1:]
            for i in range(stopping_step):
                if isinstance(self.step_stack[i], CNF):
                    x_energy = x_energy - x_energy_transforms[0]
                    x = x_steps[0]
                else:
                    x_target_energy = self.step_stack[i].target_energy(x)
                    log_quotient = x_energy - x_target_energy
                    resampling_ratio = 1.0 - torch.clip(
                        torch.exp(log_quotient - self.step_stack[i].log_c), 0, 1
                    )
                    x_energy = x_energy - torch.log(
                        1
                        - resampling_ratio
                        + self.step_stack[i].global_resampling_ratio
                    )
                x_steps = x_steps[1:]
                x_energy_transforms = x_energy_transforms[1:]
        return x_energy

    def forward(self, x, x_energy, step_index):
        # batched application of one step of the model
        step = self.step_stack[step_index]
        if isinstance(step, CNF):
            j = 0
            while j < x.shape[0]:
                batch = x[j : j + self.batch_size]
                out = step(
                    batch,
                    torch.zeros_like(x_energy[j : j + self.batch_size]),
                )
                x[j : j + self.batch_size] = out[0]
                x_energy[j : j + self.batch_size] = (
                    x_energy[j : j + self.batch_size] + out[2].squeeze()
                )
                j = j + self.batch_size
        elif isinstance(step, RejectionStep):
            j = 0
            while j < x.shape[0]:
                out = step(
                    x[j : j + self.batch_size],
                    x_energy[j : j + self.batch_size],
                )
                x[j : j + self.batch_size] = out[0]
                x_energy[j : j + self.batch_size] = out[1].reshape(
                    x_energy[j : j + self.batch_size].shape
                )
                j += self.batch_size
        else:
            raise NameError
        return x, x_energy

    def add_step(self, step):
        # add step add the end of the model
        self.step_stack.append(step)
        self.sample_stack.append(
            torch.zeros((0, self.dim), dtype=torch.float, device=device)
        )
        self.sample_energy_stack.append(
            torch.zeros((0,), dtype=torch.float, device=device)
        )
        self.add_module("Step_" + str(len(self.step_stack) - 1), self.step_stack[-1])


class RejectionStep(torch.nn.Module):
    def __init__(
        self,
        model_sampler,
        stopping_step,
        target_energy,
        resampling_rate=None,
        log_c=0.0,
    ):
        super(RejectionStep, self).__init__()
        self.target_energy = target_energy
        self.model_sampler = model_sampler
        self.stopping_step = stopping_step
        self.log_c = torch.nn.Parameter(
            torch.tensor(log_c, dtype=torch.float, device=device), requires_grad=False
        )
        self.global_resampling_ratio = torch.nn.Parameter(
            torch.tensor(0.0, dtype=torch.float, device=device), requires_grad=False
        )

        self.resampling_rate = resampling_rate

    def forward(self, x, x_energy, training=False, **kwargs):
        # x_energy=-log(p(x)), where p density of x
        x_target_energy = self.target_energy(x).squeeze()
        if len(x_target_energy.shape) == 0:
            x_target_energy = x_target_energy[None]

        log_quotient = x_energy - x_target_energy
        if self.resampling_rate is not None and training:
            self.log_c.data = torch.mean(log_quotient)
        # resampling ratio
        compute_resampling_ratio = lambda lq, log_c: 1.0 - torch.clip(
            torch.exp(lq - log_c), 0, 1
        )

        resampling_ratio = compute_resampling_ratio(log_quotient, self.log_c)
        if self.resampling_rate is not None and training:
            # bisection search for choosing c
            c_low = self.log_c.data
            ratio_low = 0.0
            c_high = self.log_c.data
            ratio_high = torch.mean(resampling_ratio)
            ratio_low = ratio_high.clone()
            it = 0
            while ratio_high < self.resampling_rate:
                c_low = c_high
                ratio_low = ratio_high
                c_high = c_high + np.log(2) * it
                resampling_ratio = compute_resampling_ratio(log_quotient, c_high)
                ratio_high = torch.mean(resampling_ratio)
                it += 1
                if it > 10000:
                    raise ValueError("Maximum number of iterations reached...")
            while ratio_low > self.resampling_rate:
                c_high = c_low
                ratio_high = ratio_low
                c_low = c_high - np.log(2) * it
                resampling_ratio = compute_resampling_ratio(log_quotient, c_low)
                ratio_low = torch.mean(resampling_ratio)
                it += 1
                if it > 10000:
                    raise ValueError("Maximum number of iterations reached...")
            c_mid = c_high
            while ratio_high - ratio_low > 1e-4:
                c_mid = 0.5 * (c_low + c_high)
                resampling_ratio = compute_resampling_ratio(log_quotient, c_mid)
                ratio_mid = torch.mean(resampling_ratio)
                if ratio_mid < self.resampling_rate:
                    c_low = c_mid
                    ratio_low = ratio_mid
                else:
                    c_high = c_mid
                    ratio_high = ratio_mid
            self.log_c.data = c_mid

        if training:
            self.global_resampling_ratio.data = torch.mean(resampling_ratio)
        resampling = torch.rand_like(resampling_ratio) <= resampling_ratio
        self.resampling = resampling

        if torch.sum(resampling.float()) > 0.5:
            # if resampling is required
            x[resampling], x_energy[resampling] = self.model_sampler(
                x[resampling].shape[0], stopping_step=self.stopping_step
            )
            x_target_energy[resampling] = self.target_energy(x[resampling]).squeeze(-1)
            log_quotient_resampled = x_energy - x_target_energy
            resampling_ratio = compute_resampling_ratio(
                log_quotient_resampled, self.log_c
            )

            # energy update
            x_energy = x_energy - torch.log(
                1 - resampling_ratio + self.global_resampling_ratio
            )
        else:
            # if no point is resampled
            x_energy = x_energy - torch.log(
                1 - resampling_ratio + self.global_resampling_ratio
            )
        return x, x_energy
