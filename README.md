# Importance Corrected Neural JKO Sampling

This repository contains the code for the paper "Importance Corrected Neural JKO Sampling" available at  
http://arxiv.org/abs/xxxx.xxxxx  
Please cite the paper, if you use the code.

The code reproduces the examples from the paper. To train the models run

```
python train_example.py --problem problem_name
```

where `problem_name` is replaced by the name of the example (choices: `mustache`, `funnel`, `8modes` , `8peaky`, `GMM10`, `GMM20`, `GMM50`, `GMM100`, `GMM200`).
To train (and save) several models on the same problem an argument `--experiment_id i` can be added, which also saves the (randomly generated) means for the `GMM` examples.

Once the models are trained, they can be loaded and evaluated by

```
python eval_example.py --problem problem_name (--experiment_id i)
```
The repository already contains the weights for `mustache`, `funnel`, `8modes`, `8peaky`, `GMM10`, `GMM20`, `GMM50` and `GMM100` with `experiment_id=1`.

For questions or comments, feel free to contact us via (contact details can be found [here](https://johertrich.github.io)).

## Setup

We ran our experiments on a single NVIDIA RTX 4090 GPU with 24 GB memory. We used `pytorch` 
with version 2.3.1 and `numpy` with version 1.26.0.
For the implementation of the continuous normalizing flows we used `torchdiffeq` with version 0.2.4, see https://github.com/rtqichen/torchdiffeq for installation instructions.

## Citation

```
@article{neuralJKO_ic,
  title={Importance Corrected Neural {JKO} Sampling},
  author={Johannes Hertrich and Robert Gruhlke},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```
