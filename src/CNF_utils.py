# This file contains a basic implementation of continuous normalizing flows.
# The code is mainly taken and adapted from
# https://github.com/rtqichen/ffjord

import torch
from torchdiffeq import odeint_adjoint as odeint

device = "cuda" if torch.cuda.is_available() else "cpu"


def sample_rademacher_like(y):
    # from ffjord
    return (
        torch.randint(low=0, high=2, size=y.shape, device=device, dtype=torch.float) * 2
        - 1
    )


def sample_rademacher(out_shape):
    # from ffjord
    return (
        torch.randint(low=0, high=2, size=out_shape, device=device, dtype=torch.float)
        * 2
        - 1
    )


def divergence_bf(dx, y, **unused_kwargs):
    # adjusted from ffjord (retain_graph instead of create_graph), we never need gradients of this function
    diags = torch.stack(
        [
            torch.autograd.grad(dx[:, i].sum(), y, retain_graph=True)[0][:, i]
            for i in range(y.shape[1])
        ],
        0,
    ).contiguous()
    return torch.sum(diags, 0)


def divergence_approx(f, y, e, training):
    # from ffjord
    e_dzdx = torch.autograd.grad(
        torch.sum(f * e), y, create_graph=training, retain_graph=True
    )[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


class DenseODENet(torch.nn.Module):
    # simplified from ffjord
    def __init__(self, input_size, activation=torch.nn.ELU(), hidden_dims=[64, 64, 64]):
        super(DenseODENet, self).__init__()
        self.layers = []
        in_size = input_size
        for dim in hidden_dims:
            layer = torch.nn.Linear(in_size + 1, dim)
            in_size = dim
            self.layers.append(layer)
        layer = torch.nn.Linear(in_size + 1, input_size)
        self.layers.append(layer)
        for i, layer in enumerate(self.layers):
            self.add_module("Layer_" + str(i), layer)
        self.activation = activation
        torch.nn.init.zeros_(self.layers[-1].weight)
        torch.nn.init.zeros_(self.layers[-1].bias)

    def forward(self, t, x):
        while len(x.shape) > 2:
            assert x.shape[2] == 1
            x = torch.sum(x, 2)
        for layer in self.layers[:-1]:
            x = torch.cat(
                (x, t * torch.ones((x.shape[0], 1), device=device, dtype=torch.float)),
                1,
            )
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](
            torch.cat(
                (x, t * torch.ones((x.shape[0], 1), device=device, dtype=torch.float)),
                1,
            )
        )
        return x


class ODEfunc(torch.nn.Module):
    # adapted from ffjord

    def __init__(
        self,
        diffeq,
        divergence_fn="approximate",
    ):
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
        self.diffeq = diffeq
        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx
        self.register_buffer("_num_evals", torch.tensor(0.0))
        self.exact = False
        self.hut_num = 5

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 3
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        if isinstance(t, torch.Tensor):
            t = t.clone().detach().type_as(y)
        else:
            t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None or self.exact:
            # self._e = torch.randn_like(y)
            if not self.exact:
                self._e = sample_rademacher_like(y)
            else:
                # self._e = torch.stack([torch.randn_like(y) for _ in range(20)],0)
                self._e = sample_rademacher((self.hut_num, y.shape[0], y.shape[1]))

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[3:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y)
            if self.exact and dy.view(dy.shape[0], -1).shape[1] <= self.hut_num:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            elif self.exact:
                divergences = torch.stack(
                    [
                        self.divergence_fn(
                            dy, y, e=self._e[i], training=self.training
                        ).view(batchsize, 1)
                        for i in range(self._e.shape[0])
                    ],
                    0,
                )
                divergence = torch.mean(divergences, 0)
            else:
                divergence = self.divergence_fn(
                    dy, y, e=self._e, training=self.training
                ).view(batchsize, 1)
            loss_fn = torch.sum(dy.view(batchsize, -1) ** 2, -1, keepdim=True)
        return tuple([dy, loss_fn, divergence])


class CNF(torch.nn.Module):
    # adapted from ffjord
    def __init__(
        self,
        odefunc,
        T=1.0,
        solver="dopri5",
        atol=1e-5,
        rtol=1e-5,
        solver_options={},
    ):
        super(CNF, self).__init__()
        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.T = torch.tensor(T, device=device)
        self.solver_options = solver_options

    def forward(self, z, energies, reverse=False):
        integration_times = torch.tensor(
            [0.0, self.T], device=device, dtype=torch.float
        )
        if reverse:
            integration_times = torch.tensor(
                [self.T, 0.0], device=device, dtype=torch.float
            )

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()
        zero = torch.zeros((z.shape[0], 1), device=device, dtype=torch.float)
        if energies is None:
            energies = zero
        else:
            energies = energies.reshape(z.shape[0], 1)
        init = (z, zero, energies)

        state_t = odeint(
            self.odefunc,
            init,
            integration_times,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
            options=self.solver_options,
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, loss_fn, logdet = state_t[:3]
        return z_t, loss_fn * self.T, logdet

    def num_evals(self):
        return self.odefunc._num_evals.item()
