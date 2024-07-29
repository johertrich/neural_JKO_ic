# Helper functions for loading and evaluating models which have been saved
# from previous training runs.

from src.CNF_utils import CNF, DenseODENet, ODEfunc
from src.model import FlowModel, RejectionStep
from src.metrics import sliced_energy_distance, log_Z_estimator, compute_mode_weights
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(
    target_energy,
    dim,
    args,
    problem_name,
    experiment_id=None,
):
    no_rejection = args.no_rejection
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else 2 * dim + 50
    n_layers = args.n_layers
    find_latent_mean = args.find_latent_mean

    name_add = "_no_rejection" if no_rejection else ""
    if experiment_id is None:
        load_dict = torch.load(
            "weights/weights_for_" + problem_name + "_" + str(dim) + name_add + ".pt"
        )
    else:
        load_dict = torch.load(
            "weights/weights_for_"
            + problem_name
            + "_"
            + str(dim)
            + "_"
            + str(experiment_id)
            + name_add
            + ".pt"
        )
    structure = load_dict["structure"]
    step_sizes = load_dict["step_sizes"]
    states = load_dict["states"]

    def get_network():
        hidden_dims = tuple([hidden_dim] * n_layers)
        diffeq = DenseODENet(
            hidden_dims=hidden_dims,
            input_size=dim,
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
            divergence_fn="approximate",
        )
        model = CNF(
            odefunc=odefunc,
            solver="rk4",
            atol=1e-3,
            rtol=1e-3,
            T=1.0,
            solver_options={},
        )
        return model

    my_mean = None
    if find_latent_mean:
        my_mean = torch.zeros(
            (1, dim), dtype=torch.float, device=device, requires_grad=True
        )
        for _ in range(1000):
            mean_ener = target_energy(my_mean).squeeze()
            mean_grad = torch.autograd.grad(mean_ener, my_mean)[0]
            my_mean = my_mean - 0.01 * mean_grad.detach()
        my_mean = my_mean.detach().squeeze()
        print(my_mean)
    networks = FlowModel(dim, latent_mean=my_mean)

    for i in range(len(structure)):
        if structure[i] == 0:
            model = get_network().to(device)
            model.eval()
            model.solver_options = dict(step_size=step_sizes[i])
            model.odefunc.exact = True
            for p in model.parameters():
                p.requires_grad_(False)
            networks.add_step(model)
        if structure[i] == 1:
            rej_step = RejectionStep(
                networks.sample,
                len(networks.step_stack),
                target_energy,
                resampling_rate=0.2,  # argument does not matter since c is loaded from checkpoint
            )
            networks.add_step(rej_step)
    networks.load_state_dict(states)
    return networks


def eval_model(model, target_energy, sampler=None, means=None, stack_size=25000):

    n_samples = 50000
    if sampler is not None:
        gt_samples = sampler(n_samples)

    model.stack_size = stack_size

    with torch.no_grad():
        tic = time.time()
        points, points_energy = model.sample(n_samples)
        toc = time.time() - tic
    sampling_time = toc

    log_Z = log_Z_estimator(points, points_energy, target_energy).detach().cpu().numpy()

    energy_distance = None
    if sampler is not None:
        with torch.no_grad():
            energy_distance = (
                sliced_energy_distance(points[:50000], gt_samples, 1000)
                .detach()
                .cpu()
                .numpy()
            )

    mode_MSE = None
    if means is not None:
        mode_weights = compute_mode_weights(points, means)
        mode_MSE = torch.linalg.norm(torch.tensor(mode_weights) - 1.0 / len(means)) ** 2
        mode_MSE = mode_MSE.detach().cpu().numpy()

    return log_Z, energy_distance, mode_MSE, sampling_time
