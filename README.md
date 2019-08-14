# SubSVDD Evaluation

This repository contains scripts and notebooks to reproduce the experiments and analyses of the paper

> Holger Trittenbach, Klemens Böhm and Ira Assent. "Active Learning of SVDD Hyperparameter Values."

For more information about this research project, see also the [lama project website](https://www.ipd.kit.edu/mitarbeiter/lama/).

## Quick Start

The analysis and main results of the experiments can be found under [notebooks](https://github.com/holtri/lama/tree/master/notebooks):
  * `dataset-overview.ipynb`: overview on data sets
  * `join_experiment_results.ipynb`: aggregation of experimental results from individual result files
  * `paper-illustrations.ipynb`: Figure 1
  * `result-analysis.ipynb`: Table 1, Figure 2

To execute the notebooks, make sure you follow the [setup section](#setup), and [download the raw results](https://www.ipd.kit.edu/mitarbeiter/lama/output.zip) into `data/output/`.

## Prerequisites

The experiments are implemented in [Julia](https://julialang.org/), some of the evaluation notebooks are written in python.
This repository contains code to setup the experiments, to execute them, and to analyze the results.

### Setup

Just clone the repo.
```bash
$ git clone https://github.com/holtri/lama.git
```
* Experiments require Julia 1.1, requirements are defined in `Manifest.toml`. To instantiate, start julia in the `lama` directory with `julia --project=. ` and run `julia> ]instantiate`. See [Julia documentation](https://docs.julialang.org/en/v1.0/stdlib/Pkg/#Using-someone-else's-project-1) for general information on how to setup this project.
* Notebooks require
  * Julia 1.1
  * python 3.7: `pands`, `numpy`, `seaborn`

### Repo Overview

* `data`
  * `input`
    * `dami-base-processed-2000`: output directory of _preprocess_data.jl_
  * `output`: output directory of experiments; _generate_experiments.jl_ creates the folder structure and experiments; _run_experiments.jl_ writes results and log files
* `notebooks`: jupyter notebooks to analyze experimental results
  * `dataset-overview.ipynb`: overview on data sets
  * `join_experiment_results.ipynb`: aggregation of experimental results from individual result files
  * `paper-illustrations.ipynb`: Figure 1
  * `result-analysis.ipynb`: Table 1, Figure 2
* `config`: configuration files for experiments
    * `config.jl`: high-level configuration
    * `evaluation-AL.jl`, `evaluation-AL-kappa.jl`, `evaluation-competitors.jl`: experiment configs
* `scripts`
  * `preprocess_data.jl`: preprocess data files into common format
  * `generate_experiments.jl`: generates experiments
  * `reduce_results.jl`: reduces result json files to single result csv
  * `run_experiments`: executes experiments

## Overview

Each step of the experiments can be reproduced, from the raw data files to the final plots that are presented in the paper.
The experiment is a pipeline of several dependent processing steps.
Each of the steps can be executed standalone, and takes a well-defined input, and produces a specified output.
The Section [Experiment Pipeline](#experiment-pipeline) describes each of the process steps.

Running the benchmark is compute intensive and takes many CPU hours.
Therefore, we also provide the [results to download](https://www.ipd.kit.edu/mitarbeiter/lama/output.zip) (51 MB).
This allows to analyze the results in the notebooks without having to run the whole pipeline.

The code is licensed under a [MIT License](https://github.com/kit-dbis/ocal-evaluation/blob/master/LICENSE.md) and the result data under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
If you use this code or data set in your scientific work, please reference the companion paper.

## Experiment Pipeline

The benchmark pipeline uses config files to set paths and experiment parameters.
There are two types of config files:
* `scripts/config.jl`: this config defines high-level information on the experiment, such as where the data files are located
* `scripts/<evaluation-AL|evaluation-AL-kappa|evaluation-competitors>.jl`: These config files define the experimental grid, including the data sets, classifiers, and kernel learning strategies.

1. _Data Preprocessing_: The preprocessing step transforms publicly available benchmark data sets into a common csv format, and subsamples large data sets to 2000 observations.
   * **Input:** Download [semantic.tar.gz](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/semantic.tar.gz) and [literature.tar.gz](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/literature.tar.gz) containing the .arff files from the DAMI benchmark repository and extract into `data/input/raw/<data set>` (e.g. `data/input/raw/Annthyroid/`).
   * **Execution:**
   ```bash
      $ julia --project="." preprocess_data.jl <config.jl>
   ```
   * **Output:** .csv files in `data/input/dami-base-processed-2000`

   We also provide our [preprocessed data to download](https://www.ipd.kit.edu/mitarbeiter/lama/input.zip) (3 MB).

2. _Generate Experiments_: This step creates a set of experiments. These specific combinations are created as a cross product of the vectors in the config file that is passed as an argument.
   * **Input**: Full path to config file `<config_file.jl>` (e.g., config/evaluation-AL.jl), preprocessed data files
   * **Execution:**
   ```bash
    $ julia --project="." generate_experiments.jl <config_file.jl>
   ```
   * **Output:**
     * Creates an experiment directory with the naming `<exp_name>`. The directories created contains several items:
       * `log` directory: skeleton for experiment logs (one file per experiment), and worker logs (one file per worker)
       * `results` directory: skeleton for result files
       * `experiments.jser`: this contains a serialized Julia Array with experiments. Each experiment is a Dict that contains the specific combination. Each experiment can be identified by a unique hash value.
       * `experiment_hashes`: file that contains the hash values of the experiments stored in `experiments.jser`
       * `generate_experiments.jl`: a copy of the file that generated the experiments
       * `config.jl`: a copy of the config file used to generate the experiments

3. _Run Experiments_: This step executes the experiments created in Step 2.
Each experiment is executed on a worker. In the default configuration, a worker is one process on the localhost.
A worker takes one specific configuration, runs the active learning experiment, and writes result and log files.
  * **Input:** Generated experiments from step 2.
  * **Execution:**
  ```bash
     $ julia --project="." run_experiments.jl /full/path/to/ocal-evaluation/scripts/config.jl
  ```
  * **Output:** The output files are named by the experiment hash
    * Experiment log (e.g., `data/output/evaluation-AL/results/log/476519099826377054.log`)
    * Result .json file (e.g., `data/output/evaluation-AL/results/Annthyroid/Annthyroid_withoutdupl_norm_07_ALKernel_nns=1_adj=true_476519099826377054`)

4. _Reduce Results_: `join_experiment_results.ipynb` in the `notebooks` directory merges experiment directories into .pkl files

5. _Analyze Results:_ `result-analysis.ipynb` in the `notebooks` directory contains code to analyze results, and create plots and tables

## Authors
We welcome contributions and bug reports.

This package is developed and maintained by [Holger Trittenbach](https://github.com/holtri/)
