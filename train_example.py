import argparse
from src.train import train
from examples.configs import get_config
from examples.energies import get_problem

parser = argparse.ArgumentParser(description="Choosing problem")
parser.add_argument("--problem", type=str, default="funnel")
parser.add_argument(
    "--experiment_id", type=int, default=None
)  # numbering for several runs, can be None
inp = parser.parse_args()

experiment_id = inp.experiment_id

# specify problem and dimension
if inp.problem == "funnel":
    dim = 10
    problem_name = "funnel"
elif inp.problem == "mustache" or inp.problem == "schnauzbart":
    dim = 2
    problem_name = "schnauzbart"
elif inp.problem == "8peaky":
    dim = 2
    problem_name = "8peaky"
elif inp.problem == "8mixtures" or inp.problem == "8modes":
    dim = 2
    problem_name = "8mixtures"
elif inp.problem == "GMM10":
    dim = 10
    problem_name = "mixtures"
elif inp.problem == "GMM20":
    dim = 20
    problem_name = "mixtures"
elif inp.problem == "GMM50":
    dim = 50
    problem_name = "mixtures"
elif inp.problem == "GMM100":
    dim = 100
    problem_name = "mixtures"
elif inp.problem == "GMM200":
    dim = 200
    problem_name = "mixtures"
elif inp.problem == "lgcp":
    dim = 1600
    problem_name = "lgcp"


# make choice of the means reproducible if this is an mixture example
# otherwise this argument will be ignored
additional_info = {}
if experiment_id is not None:
    additional_info["mean_id"] = experiment_id

# load target energy (energy = negative log density up to some constant)
target_energy, sampler, dim, axis_scale, additional_info = get_problem(
    problem_name, dim=dim, additional_info=additional_info
)

# load means of mixture modes for evaluation if available
means = additional_info["means"] if "means" in additional_info.keys() else None

# load hyperparameters
args = get_config(problem_name, dim)

# training
train(
    target_energy,
    dim,
    args,
    sampler=sampler,  # required for computing the energy distance; can be None
    means=means,  # required for computing the mode MSE; can be None
    experiment_id=experiment_id,  # used for running several experiments; can be None
    problem_name=problem_name,  # defines path to save the weights
)
