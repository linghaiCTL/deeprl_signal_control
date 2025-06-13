# Deep RL for traffic signal control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repo if forked from https://github.com/cts198859/deeprl_signal_control.
The checkpoints and training log are saved in ./output


Available algorithms:
IQL, IA2C, IA2C with stabilization. IPPO modified from IA2C, IC3NET and IC3NET with attention mechanism.

## Requirements
* Python3==3.5
* [Tensorflow](http://www.tensorflow.org/install)==1.12.0
* [SUMO](http://sumo.dlr.de/wiki/Installing)>=1.1.0
* torch

You can simply run setup_ubuntu.sh to install the packages

## Usages
First define all hyperparameters in a config file under `[config_dir]`, and create the base directory of experiements `[base_dir]`. Before training, please call `build_file.py` under `[environment_dir]/data/` to generate SUMO network files for `small_grid` and `large_grid` environments.

## To run train and evaluate, you can simply modify and run run.sh


1. To train the agent from original repo, run
~~~
python3 main.py --base-dir [base_dir]/[agent] train --config-dir [config_dir] --test-mode no_test
~~~
`[agent]` is from `{ia2c, ma2c}`. `no_test` is suggested, since tests will significantly slow down the training speed.
~~~

To train the new agent (ippo ic3net, ic3netattn), run
~~~
python3 main_torch.py --base-dir [base_dir]/[agent] train --config-dir [config_dir] --test-mode no_test
~~~
`[agent]` is from `{ippo, ic3net, ic3netattn}`. `no_test` is suggested, since tests will significantly slow down the training speed.
~~~

