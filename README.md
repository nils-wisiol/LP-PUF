# LP-PUF: Towards Attack Resilient Arbiter PUF-Based Strong PUFs

This repository contains simulation and analysis code for the LP-PUF.
The current version of LP-PUF is 1 to allow for future improvements.

## How to Use this Repository

All code in this repository uses [pypuf](//github.com/nils-wisiol/pypuf), a PUF cryptanalysis tool.
It can be installed using

```shell
python3 -m pip install pypuf
```

Furthermore, [pandas](https://pandas.pydata.org/) and [seaborn](https://seaborn.pydata.org/) are used for data analysis
and visualization. Some code is organized in traditional Python modules, most analyses are run in
[Juypter notebooks](https://jupyter.org/).


## Simulation of the LP-PUF

The [simulation of the LP-PUF](lppuf.py) is based on the
[Arbiter PUF simulation](https://pypuf.readthedocs.io/en/latest/simulation/delay.html) of pypuf.
An LP-PUF simulation instance can be created by passing the relevant security parameters and a `seed` which is used
to initialize the PRNG to obtain "physical" intrinsic parameters of the involved Arbiter PUFs:

```python3
import lppuf
my_lp_puf = lppuf.LPPUFv1(n=64, m=8, seed=1)
```

where `n` specifies the challenge length, and `m` defines the number of Arbiter PUFs in the first layer.
Additionally, the parameters `noisiness_1` and `noisiness_2` may be given by non-negative floats to control the
reliability of the first and third layer, respectively.
The defined PUF instances can be evaluated on challenges like so:

```python3
import pypuf.io
my_lp_puf.eval(pypuf.io.random_inputs(n=64, N=3, seed=1))
```


## PUF Metrics Analysis

The LP-PUF is analyzed for its [bias](LPPUFv1%20Bias.ipynb), [reliability](LPPUFv1%20Reliability.ipynb),
[uniqueness](LPPUFv1%20Uniqueness.ipynb), and [bit sensitivity](LPPUFv1%20Bit%20Sensitivity.ipynb) in the
corresponding notebook files.


## Security Analysis

### Logistic Regression / Splitting Attack and MLP Attack

The security analysis with respect to the Logistic Regression / Splitting Attack and the MLP attack split into two
parts.
In the first part, the attacks are run on LP-PUF simulations.
In the second part, the results are analyzed using a Jupyter notebook.
This split was done as the attacks require a relatively long time to complete and are preferably run on a SLURM
computing cluster.

To run the attacks on a single computer, use

```shell
python3 -m lppufv1_lr_full results/v1/lr 0 1  # for the Logistic Regression / Splitting Attack
python3 -m lppufv1_mlp_full results/v1/mlp 0 1  # for the MLP attack
```

The results will be stored at `results/v1/lr` and `results/v1/mlp`, respectively.
Afterwards, the analysis can be done by running the [LR Attack](LPPUFv1%20LR%20Attack.ipynb) and
[MLP Attack](LPPUFv1%20MLP%20Attack.ipynb) notebooks.

The results are not contained in this repository due to their large file size.
However, the notebooks contain the analysis results.

### Reliability Attack

As the analysis for reliability-based attacks is done on reliability values rather than by running the actual attack,
the run time is relatively short and the whole analysis can be done within a Jupyter notebook.
There are analyses for [Layer 1](LPPUFv1%20Reliability%20Correlation%20Layer%201.ipynb) and
[Layer 3](LPPUFv1%20Reliability%20Correlation%20Layer%203.ipynb)

## Extending and Contributing

All source code in this repository is licensed under the GPLv3. Everything else is licenced as
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
([full license](https://creativecommons.org/licenses/by/4.0/legalcode)).
