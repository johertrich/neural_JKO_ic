# In this file we define the target energies (= negative log density up to constant)

import torch
import math
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import os
from examples.lgcp_energy import log_gaussian_cox_process
import pickle
from sympy import symbols, expand, poly
from sympy.matrices import Matrix

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_problem(key, dim=None, additional_info={}):
    # selects problem
    axis = None
    if key == "schnauzbart":
        C0, sampler, const = schnauzbart(2)
        C0 = C0.to(device).to(torch.float32)

        def target_energy(x):
            X = torch.stack([x[:, 0] ** i for i in range(8 + 1)], dim=1)
            Y = torch.stack([x[:, 1] ** j for j in range(2 + 1)], dim=1)
            return torch.sum(X * (Y @ C0.T), -1) + const

        dim = 2
        axis = [[-5.0, 5.0], [-3, 10]]

    elif key == "8mixtures":
        target_energy, sampler, means = target_energy_8mixtures()
        dim = 2
        axis = [[-2.3, 0.3], [-1.3, 1.3]]
        additional_info["means"] = means
    elif key == "8peaky":
        target_energy, sampler, means = target_energy_8mixtures(sigma_sq=0.005)
        dim = 2
        axis = [[-2.3, 0.3], [-1.3, 1.3]]
        additional_info["means"] = means
    elif key == "mixtures":
        dim = 10 if dim is None else dim
        means = additional_info["means"] if "means" in additional_info.keys() else None
        mean_id = (
            additional_info["mean_id"] if "mean_id" in additional_info.keys() else None
        )
        target_energy, sampler, means = target_energy_mixtures(
            dim=dim, means=means, mean_id=mean_id
        )
        axis = [[-1.3, 1.3], [-1.3, 1.3]]
        additional_info["means"] = means
    elif key == "funnel":
        dim = 10 if dim is None else dim
        target_energy, sampler = funnel_guy(dim)
        axis = [[-15, 10], [-20.0, 20.0]]
    elif key == "lgcp":
        dim = 1600
        target_energy = log_gaussian_cox_process()
        sampler = None
        axis = [[0, 10], [0, 10.0]]
    return target_energy, sampler, dim, axis, additional_info


def schnauzbart(d=2):
    assert d == 2

    assert d > 1

    param = 0.9

    Sigma = Matrix([[1, param], [param, 1]])
    Sigma.inv()
    x, y = symbols("x y")
    f = lambda X, Y: -0.5 * Matrix([[X], [Y]]).T @ Sigma.inv() @ Matrix([[X], [Y]])
    T = lambda X, Y: Matrix([[X], [(Y - (X**2 - 1) ** 2)]])
    T_expand = expand(T(x, y))

    f_expand = expand(f(T_expand[0, 0], T_expand[1, 0]))
    p = poly(f_expand[0, 0])

    def initial_cond(degrees=[8, 2]):
        C0 = torch.zeros(degrees[0] + 1, degrees[1] + 1, dtype=torch.double)
        for i in range(8 + 1):
            for j in range(2 + 1):
                C0[i, j] = float(p.coeff_monomial(x**i * y**j))
        return C0

    C0 = -initial_cond()

    Sigma = torch.tensor([[1, param], [param, 1]])
    normal_dist = MultivariateNormal(torch.zeros(d), covariance_matrix=Sigma)
    transport = lambda x: torch.stack(
        [x[:, 0], x[:, 1] - (x[:, 0] ** 2 - 1) ** 2], dim=1
    )

    def sampler(N):
        return -transport(normal_dist.sample((N,))).to(device)

    const = np.log((2 * np.pi) * np.sqrt(np.linalg.det([[1, param], [param, 1]])))
    return C0, sampler, const


def target_energy_mixtures(K=10, sigma_sq=0.01, dim=10, means=None, mean_id=None):
    if not os.path.isdir("data"):
        os.mkdir("data")
    if not os.path.isdir("data/mixture_means"):
        os.mkdir("data/mixture_means")
    if means is None:
        fname = (
            "data/mixture_means/mixture_means" + str(mean_id) + "_" + str(dim) + ".pck"
        )
        if mean_id is None or not os.path.isfile(fname):
            means = []
            for k in range(K):
                means.append(
                    2 * torch.rand((dim,), dtype=torch.float, device=device) - 1
                )
            if mean_id is not None:
                with open(fname, "wb") as f:
                    pickle.dump(
                        {"means": [mean.detach().cpu().numpy() for mean in means]}, f
                    )
        else:
            with open(fname, "rb") as f:
                dict_pck = pickle.load(f)
            means = [
                torch.tensor(mean, dtype=torch.float, device=device)
                for mean in dict_pck["means"]
            ]

    mvns = []
    cov = np.sqrt(sigma_sq) * torch.eye(dim, device=device)
    for k in range(K):
        mvns.append(MultivariateNormal(means[k], scale_tril=cov))

    def target_energy(x):
        log_probs = torch.stack([mvn.log_prob(x) - np.log(1.0 * K) for mvn in mvns])
        return -torch.logsumexp(log_probs, 0)

    def sampler(N):
        select_class = torch.randint(0, K, size=(N,), device=device)
        samples = torch.zeros((N, dim), device=device)
        for k in range(K):
            correct_class = select_class == k
            num_class = torch.sum(correct_class.to(torch.int32))
            samples[correct_class] = mvns[k].sample((num_class,))
        return samples

    return target_energy, sampler, means


def target_energy_8mixtures(sigma_sq=0.01):
    K = 8
    dim = 2
    means = []
    mvns = []
    cov = np.sqrt(sigma_sq) * torch.eye(dim, device=device)
    shift = 1.0
    for k in range(K):
        means.append(
            torch.tensor(
                [np.cos(2 * math.pi * k / K) - shift, np.sin(2 * math.pi * k / K)],
                dtype=torch.float,
                device=device,
            )
        )
        mvns.append(MultivariateNormal(means[-1], scale_tril=cov))

    def target_energy(x):
        log_probs = torch.stack([mvn.log_prob(x) - np.log(1.0 * K) for mvn in mvns])
        return -torch.logsumexp(log_probs, 0)

    def sampler(N):
        select_class = torch.randint(0, K, size=(N,), device=device)
        samples = torch.zeros((N, 2), device=device)
        for k in range(K):
            correct_class = select_class == k
            num_class = torch.sum(correct_class.to(torch.int32))
            samples[correct_class] = mvns[k].sample((num_class,))
        return samples.to(device)

    return target_energy, sampler, means


def funnel_guy(dim):
    var = 9.0

    def target_energy(x):
        v = x[:, 0]
        out = 0.5 / var * v**2 + 0.5 * np.log(2 * math.pi * var)
        out = (
            out
            + 0.5 * torch.exp(-v) * torch.sum(x[:, 1:] ** 2, -1)
            + 0.5 * (dim - 1) * v
            + 0.5 * (dim - 1) * np.log(2 * math.pi)
        )
        return out

    def sampler(N):
        v = torch.sqrt(torch.tensor(var)) * torch.randn(N, 1)
        x = torch.exp(v / 2) * torch.randn(N, dim - 1)
        return torch.cat([v, x], dim=1).to(device)

    return target_energy, sampler
