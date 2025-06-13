# Deep RL for traffic signal control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repo if forked from https://github.com/cts198859/deeprl_signal_control.

Available cooperation levels:
* Centralized: a global agent that makes global control w/ global observation, reward.
* Decentralized: multiple local agents that make local control independently w/ neighborhood information sharing.

Available NN layers:
Fully-connected, LSTM.

Available algorithms:
IQL, IA2C, IA2C with stabilization (called MA2C in this paper). For more advanced algorithms, please check [deeprl_network](https://github.com/cts198859/deeprl_network). IPPO modified from IA2C, IC3NET and IC3NET with attention mechanism.

## Requirements
* Python3==3.5
* [Tensorflow](http://www.tensorflow.org/install)==1.12.0
* [SUMO](http://sumo.dlr.de/wiki/Installing)>=1.1.0

Required packages can be installed by running `setup_mac.sh` or `setup_ubuntu.sh`. 

Attention: the code on master branch is for SUMO version >= 1.1.0. Please go to branch [sumo-0.32.0](https://github.com/cts198859/deeprl_signal_control/tree/sumo-0.32.0) if you are using the old SUMO version.

## Usages
First define all hyperparameters in a config file under `[config_dir]`, and create the base directory of experiements `[base_dir]`. Before training, please call `build_file.py` under `[environment_dir]/data/` to generate SUMO network files for `small_grid` and `large_grid` environments.

1. To train a new agent, run
~~~
python3 main.py --base-dir [base_dir]/[agent] train --config-dir [config_dir] --test-mode no_test
~~~
`[agent]` is from `{ia2c, ma2c, ippo, ic3net, ic3netattn}`. `no_test` is suggested, since tests will significantly slow down the training speed.
~~~
