import argparse


def get_config(problem=None, dim=None):
    args = argparse.Namespace()

    # Parameters for the velocity field of the CNF
    args.n_layers = 3  # number of layers
    args.hidden_dim = None  # number hidden neurons (None means 2*dim+50)

    # Training parmeters for the CNF in the neural JKO scheme
    args.lr = 5e-3  # learning rate for training the CNFs in the neural JKO steps
    args.lr_decay = 0.95  # ratio to reduce the learning rate after each 100 steps
    args.num_steps_nf = 5000  # number of training steps
    args.batch_size = 5000  # batch size

    # Parameters of the rejection steps
    args.rejection_steps_per_block = (
        3  # number of rejection steps following a neural JKO step
    )
    args.resampling_rate = 0.2  # ratio of samples which are resampled (used for select c in the rejection steps)
    args.no_rejection = False  # set True for skipping the rejection steps (i.e. to run the neural JKO scheme without rejection steps)

    # Structure of the scheme
    args.n_rejection_steps = 6  # number of rejection blocks
    args.n_flow_steps = 4  # number of flow steps before the first rejection block. None chooses this adaptively
    args.initial_tau = 0.01  # initial choice of tau for the neural JKO steps (tau = step size in the JKO scheme)
    args.step_increase_base = 4  # multiply tau by this value after each neural JKO step

    # Other parameters
    args.n_samples = (
        50000  # use this number of samples for training the normalizing flows
    )
    args.find_latent_mean = False  # True for running an optimization scheme to change the mean of the latent distribution
    args.stack_size = 25000  # stack size for parallelized sampling (no effect on the results, but possibly on the speed)
    args.verbose = True  # print more training stats...
    args.latent_scale = 1.0  # scale for the latent variable
    args.grad_flow_step = False  # True for replacing the first CNF by running a gradient flow on the negative log-energy

    if problem is None:
        print(
            "No problem was given. Use some standard parameters which might be suboptimal."
        )
        return args

    if problem == "funnel":
        args.initial_tau = 5
        args.hidden_dim = 256
        args.num_steps_nf = 10000
        args.n_samples = 200000
        args.n_flow_steps = 4
        args.lr_decay = 0.98

    if problem == "schnauzbart":
        # schnauzbart = mustache
        args.inital_tau = 0.05
        args.n_flow_steps = 6
        args.n_rejection_steps = 6

    if problem == "8mixtures":
        # 8mixtures = shifted 8 modes
        args.initial_tau = 0.05
        args.n_flow_steps = 2
        args.n_rejection_steps = 4

    if problem == "8peaky":
        # 8peaky = shifted 8 peaky
        args.n_flow_steps = 2
        args.n_rejection_steps = 4

    if problem == "mixtures":
        # mixtures = GMM-d
        assert dim in [10, 20, 50, 100, 200]
        args.initial_tau = 0.0025 if dim < 200 else 0.001
        args.hidden_dim = None if dim < 200 else 512
        args.n_flow_steps = 4 if dim < 200 else 5
        args.n_rejection_steps = 6
        if dim in [50, 100]:
            args.n_rejection_steps = 7
        if dim == 200:
            args.n_rejection_steps = 8
            args.lr = 1e-3
            args.batch_size = 2000
        args.step_increase_base = 2

    if problem == "lgcp":
        args.hidden_dim = 1024
        args.initial_tau = 5
        args.find_latent_mean = True
        args.lr = 5e-4
        args.num_steps_nf = 10000
        args.n_flow_steps = 3
        args.n_rejection_steps = 6
        args.n_samples = 100000
        args.batch_size = 500

    if problem == "GMM40":
        args.latent_scale = 40.0
        args.n_samples = 50000
        args.step_increase_base = 2.0
        args.hidden_dim = 256
        args.num_steps_nf = 2000
        args.n_flow_steps = 4
        args.n_rejection_steps = 7
        args.initial_tau = 1.0
        args.lr = 1e-3
        args.grad_flow_step = True

    return args
