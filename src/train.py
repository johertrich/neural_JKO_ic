# Training function for a importance corrected neural JKO model

from src.CNF_utils import CNF, DenseODENet, ODEfunc, GradientFlow
from src.model import FlowModel, RejectionStep
from src.metrics import sliced_energy_distance, log_Z_estimator, compute_mode_weights
import torch
import numpy as np
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(
    target_energy,
    dim,
    args,
    sampler=None,
    means=None,
    experiment_id=None,
    problem_name=None,
):
    # read hyperparameters from args
    initial_tau = args.initial_tau
    n_rejection_steps = args.n_rejection_steps
    n_layers = args.n_layers
    lr_start = args.lr
    no_rejection = args.no_rejection
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else 2 * dim + 50
    n_steps = args.num_steps_nf
    batch_size_x = args.batch_size
    step_increase_base = args.step_increase_base
    n_samples = args.n_samples
    find_latent_mean = args.find_latent_mean
    stack_size = args.stack_size
    verbose = args.verbose
    n_flow_steps = args.n_flow_steps
    lr_decay = args.lr_decay
    rejection_steps_per_block = args.rejection_steps_per_block
    resampling_rate = args.resampling_rate
    latent_scale = args.latent_scale
    grad_flow_step = args.grad_flow_step

    if not os.path.isdir("weights"):
        os.mkdir("weights")

    def target_energy_grad(x, detach=False):
        x_ = x.clone()
        x_.requires_grad_()
        with torch.set_grad_enabled(True):
            energy = target_energy(x_)
            grad = torch.autograd.grad(torch.sum(energy), x_, create_graph=not detach)[
                0
            ]
        if detach:
            grad = grad.detach()
        return grad

    def get_first_network():
        # constructor for CNF
        diffeq = GradientFlow(target_energy_grad, factor=3.0)
        odefunc = ODEfunc(
            diffeq=diffeq,
            divergence_fn="approximate",
        )
        model = CNF(
            odefunc=odefunc,
            solver="dopri5",
            atol=1e-3,
            rtol=1e-3,
            T=1.0,
            solver_options={},
        )
        return model

    def get_network(layer_size, n_layers):
        # constructor for CNF
        hidden_dims = tuple([layer_size] * n_layers)
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
            solver="dopri5",
            atol=1e-3,
            rtol=1e-3,
            T=1.0,
            solver_options={},
        )
        return model

    def loss_function(model, x, x_energy, tau):
        # loss function for a CNF model
        out = model(x, torch.zeros_like(x_energy))
        loss = torch.mean(target_energy(out[0]))
        w2_penalty = torch.mean(out[1])
        loss = loss + w2_penalty / tau
        loss = loss - torch.mean(out[2] + x_energy)
        return loss, w2_penalty

    if sampler is not None:
        gt_samples = sampler(n_samples)
        dists = []
        for _ in range(10):
            # compute reference energy distance of two sets of samples
            samples1 = sampler(n_samples)
            samples2 = sampler(n_samples)
            dists.append(
                sliced_energy_distance(samples1[:50000], samples2[:50000], 1000).item()
            )

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

    # maximal number of networks...
    n_nets = 100
    taus = [initial_tau * step_increase_base**i for i in range(n_nets)]
    networks = FlowModel(
        dim, batch_size=stack_size, latent_mean=my_mean, latent_scale=latent_scale
    )
    dataset, dataset_energy = networks.sample(n_samples)

    def print_statistics(points, points_energy):
        print(
            f"log Z estimator: {log_Z_estimator(points, points_energy, target_energy)}"
        )
        if sampler is not None:
            with torch.no_grad():
                print(
                    f"Energy distance to target: {sliced_energy_distance(points[:50000], gt_samples[:50000], 1000)} (ref {np.mean(dists)})"
                )
        if means is not None:
            mode_weights = compute_mode_weights(points, means)
            print("Mode weights:")
            print(mode_weights)
            print(
                f"Mode MSE : {torch.linalg.norm(torch.tensor(mode_weights) - 1. / len(means))**2}"
            )

    print("Initial metrics:")
    print_statistics(dataset, dataset_energy)

    def save_model(flow_model, path):
        states = flow_model.state_dict()
        structure = [layer_code(step) for step in flow_model.step_stack]
        step_sizes = [
            step.solver_options["step_size"] if isinstance(step, CNF) else None
            for step in flow_model.step_stack
        ]
        torch.save(
            dict(
                structure=structure,
                step_sizes=step_sizes,
                states=states,
            ),
            path,
        )

    def layer_code(layer):
        # specifies the kind of a layer as integer
        if isinstance(layer, CNF):
            return 0
        if isinstance(layer, RejectionStep):
            return 1
        return -1

    net_num = 0
    rejection_counter = 0
    while net_num < n_nets:
        w2_mean = 0
        fs_mean = 0
        if net_num == 0 and grad_flow_step:
            model = get_first_network()
            x = dataset[:batch_size_x]
            model(x, torch.zeros_like(dataset_energy[:batch_size_x]))
            fs_mean = model.odefunc._num_evals.item()
            model.eval()
        else:
            model = get_network(hidden_dim, n_layers).to(device)
            lr = lr_start
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            perm_ = torch.randperm(n_samples)
            start_ind = 0
            for steps in (progress_bar := tqdm(range(n_steps), disable=not verbose)):
                if (steps + 1) % 100 == 0:
                    for g in optimizer.param_groups:
                        lr = lr * lr_decay
                        g["lr"] = lr
                if start_ind + batch_size_x > n_samples:
                    start_ind = 0
                    perm_ = torch.randperm(n_samples)
                optimizer.zero_grad()
                perm = perm_[start_ind : start_ind + batch_size_x]
                start_ind += batch_size_x
                x = dataset[perm]
                x_energy = dataset_energy[perm]
                loss, w2_penalty = loss_function(model, x, x_energy, taus[net_num])
                fs_mean = 0.9 * fs_mean + 0.1 * model.odefunc._num_evals.item()
                loss.backward()
                optimizer.step()
                w2_mean = 0.9 * w2_mean + 0.1 * w2_penalty.item()
                progress_bar.set_description(
                    "Loss: {0:.4f}, W2: {1:.4f}, LR: {2:.3f}e-3, fs_mean: {3:.1f}, W2_mean={4:.4f}".format(
                        loss.item(), w2_penalty.item(), lr * 1e3, fs_mean, w2_mean
                    )
                )
            if not verbose:
                print(
                    "Loss: {0:.4f}, W2: {1:.4f}, Num evals: {2}, fs_mean: {3:.1f}, W2_mean={4:.4f}".format(
                        loss.item(),
                        w2_penalty.item(),
                        model.odefunc._num_evals.item(),
                        fs_mean,
                        w2_mean,
                    )
                )
            model.eval()
        model.solver = "rk4"
        model.solver_options = dict(step_size=1.0 / np.ceil(0.25 * fs_mean + 1))
        model.odefunc.exact = True
        for p in model.parameters():
            p.requires_grad_(False)
        networks.add_step(model)

        with torch.no_grad():
            dataset, dataset_energy = networks.sample(n_samples)

        ds_mean = torch.mean(dataset, 0)
        ds_var = torch.mean(torch.sum((dataset - ds_mean[None, :]) ** 2, -1))
        print(w2_mean * taus[0] / (0.0025 * ds_var * taus[net_num]).item())

        print("\n\nBefore rejection step:")
        print_statistics(dataset, dataset_energy)

        if n_flow_steps is None:
            do_rejection = w2_mean * taus[0] / (0.0025 * ds_var * taus[net_num]) < 0.1
        else:
            do_rejection = net_num >= n_flow_steps
        if do_rejection and no_rejection:
            rejection_counter += 1
        if do_rejection and not no_rejection:
            rejection_counter += 1
            #########################################
            # rejection step
            #########################################
            for _ in range(rejection_steps_per_block):
                rej_step = RejectionStep(
                    networks.sample,
                    len(networks.step_stack),
                    target_energy,
                    resampling_rate=resampling_rate,
                )
                with torch.no_grad():
                    dataset, dataset_energy = rej_step(
                        dataset, dataset_energy, training=True
                    )
                networks.add_step(rej_step)

            print("\nAfter rejection step:")
            print_statistics(dataset, dataset_energy)

        net_num += 1
        if problem_name is None:
            print("No problem name for saving weights defined! Do not save weights!")
        else:
            name_add = "_no_rejection" if no_rejection else ""
            if experiment_id is None:
                save_model(
                    networks,
                    "weights/weights_for_"
                    + problem_name
                    + "_"
                    + str(dim)
                    + name_add
                    + ".pt",
                )
            else:
                save_model(
                    networks,
                    "weights/weights_for_"
                    + problem_name
                    + "_"
                    + str(dim)
                    + "_"
                    + str(experiment_id)
                    + name_add
                    + ".pt",
                )
        if rejection_counter >= n_rejection_steps:
            break
