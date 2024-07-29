# This file contains the evaluation metrics, i.e., the energy distance (computed via slicing)
# the log(Z) estimator and the mode weight computation routine

import torch
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


def log_Z_estimator(x, x_energy, target_energy):
    approx_KL = torch.mean(target_energy(x) - x_energy)
    return -approx_KL


def compute_mode_weights(x, means):
    dists_ = []
    for mean in means:
        dists_.append(torch.sum((x - mean[None, :]) ** 2, -1))
    dists_ = torch.stack(dists_, -1)
    closest = torch.argmin(dists_, -1)
    mode_weights = []
    for i in range(dists_.shape[-1]):
        mode_weights.append((torch.count_nonzero(closest == i) / x.shape[0]).item())
    return mode_weights


def energy_distance_1D(x, y):
    # 1D energy distance via cdf, see Szeleky 2002
    N = x.shape[1]
    M = y.shape[1]
    points, inds = torch.sort(torch.cat((x, y), 1))
    x_where = torch.where(inds < N, 1.0, 0.0).type(torch.float)
    F_x = torch.cumsum(x_where, -1) / N
    y_where = torch.where(inds >= N, 1.0, 0.0).type(torch.float)
    F_y = torch.cumsum(y_where, -1) / M
    mmd = torch.sum(
        ((F_x[:, :-1] - F_y[:, :-1]) ** 2) * (points[:, 1:] - points[:, :-1]), -1
    )
    return mmd


def compute_sliced_factor(d):
    # Compute the slicing constant within the negative distance kernel
    k = (d - 1) // 2
    fac = 1.0
    if (d - 1) % 2 == 0:
        for j in range(1, k + 1):
            fac = 2 * fac * j / (2 * j - 1)
    else:
        for j in range(1, k + 1):
            fac = fac * (2 * j + 1) / (2 * j)
        fac = fac * math.pi / 2
    return fac


def sliced_energy_distance(x, y, n_projections, sliced_factor=None):
    # Compute energy distance via slicing
    d = x.shape[1]
    if sliced_factor is None:
        sliced_factor = compute_sliced_factor(d)
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    xi = torch.randn((n_projections, d), dtype=torch.float, device=device)
    xi = xi / torch.sqrt(torch.sum(xi**2, -1, keepdim=True))
    xi = xi.unsqueeze(1)
    x_proj = torch.nn.functional.conv1d(x.reshape(1, 1, -1), xi, stride=d).reshape(
        n_projections, -1
    )
    y_proj = torch.nn.functional.conv1d(y.reshape(1, 1, -1), xi, stride=d).reshape(
        n_projections, -1
    )
    mmds = energy_distance_1D(x_proj, y_proj)
    return torch.mean(mmds) * sliced_factor
