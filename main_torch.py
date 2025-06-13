"""
Main function for training and evaluating agents in traffic envs
@author: Tianshu Chu
"""

import argparse
import configparser
import logging
import tensorflow as tf
import threading
# from envs.test_env import GymEnv
from envs.small_grid_env import SmallGridEnv, SmallGridController
from envs.large_grid_env import LargeGridEnv, LargeGridController
from envs.real_net_env import RealNetEnv, RealNetController
from agents.models import MultiAgentPolicyPPOWrapper, MultiAgentPolicyIC3NetWrapper,MultiAgentPolicyIC3NetAttnWrapper
from utils import (Counter, TorchTrainer, TorchEvaluator,
                   check_dir, copy_file, find_file,
                   init_dir, init_log, init_test_flag,
                   plot_evaluation, plot_train)

def parse_args():
    default_base_dir = '/Users/tchu/Documents/rl_test/signal_control_results/eval_sep2019/large_grid'
    default_config_dir = './config/config_test_large.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    subparsers = parser.add_subparsers(dest='option', help="train or evaluate")
    sp = subparsers.add_parser('train', help='train a single agent under base dir')
    sp.add_argument('--test-mode', type=str, required=False,
                    default='no_test',
                    help="test mode during training",
                    choices=['no_test', 'in_train_test', 'after_train_test', 'all_test'])
    sp.add_argument('--config-dir', type=str, required=False,
                    default=default_config_dir, help="experiment config path")
    sp = subparsers.add_parser('evaluate', help="evaluate and compare agents under base dir")
    sp.add_argument('--agents', type=str, required=False,
                    default='naive', help="agent folder names for evaluation, split by ,")
    sp.add_argument('--evaluation-policy-type', type=str, required=False, default='default',
                    help="inference policy type in evaluation: default, stochastic, or deterministic")
    sp.add_argument('--evaluation-seeds', type=str, required=False,
                    default=','.join([str(i) for i in range(10000, 100001, 10000)]),
                    help="random seeds for evaluation, split by ,")
    sp.add_argument('--demo', action='store_true', help="shows SUMO gui")
    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


def init_env(config, port=0, naive_policy=False):
    if config.get('scenario') == 'small_grid':
        if not naive_policy:
            return SmallGridEnv(config, port=port)
        else:
            env = SmallGridEnv(config, port=port)
            policy = SmallGridController(env.node_names)
            return env, policy
    elif config.get('scenario') == 'large_grid':
        if not naive_policy:
            return LargeGridEnv(config, port=port)
        else:
            env = LargeGridEnv(config, port=port)
            policy = LargeGridController(env.node_names)
            return env, policy
    elif config.get('scenario') == 'real_net':
        if not naive_policy:
            return RealNetEnv(config, port=port)
        else:
            env = RealNetEnv(config, port=port)
            policy = RealNetController(env.node_names, env.nodes)
            return env, policy
    elif config.get('scenario') in ['Acrobot-v1', 'CartPole-v0', 'MountainCar-v0']:
        return GymEnv(config.get('scenario'))
    else:
        if not naive_policy:
            return None
        else:
            return None, None


def train(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    config_dir = args.config_dir
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)
    in_test, post_test = init_test_flag(args.test_mode)

    # init env
    env = init_env(config['ENV_CONFIG'])
    logging.info('Training: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r' %
                 (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls))

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init centralized or multi agent
    seed = config.getint('ENV_CONFIG', 'seed')
    if env.agent == 'ippo':
        model = MultiAgentPolicyPPOWrapper(env.n_s_ls, env.n_a_ls, env.n_w_ls, config['MODEL_CONFIG'])
        print("Using policy IPPO")
    elif env.agent == 'ic3net':
        model = MultiAgentPolicyIC3NetWrapper(env.n_s_ls, env.n_a_ls, env.n_w_ls, config['MODEL_CONFIG'])
        print("Using policy IC3Net")
    elif env.agent == 'ic3netattn':
        model = MultiAgentPolicyIC3NetAttnWrapper(env.n_s_ls, env.n_a_ls, env.n_w_ls, config['MODEL_CONFIG'])
        print("Using policy IC3Net with Attention")
    else:
        print('Not support torch yet')

    summary_writer = tf.summary.FileWriter(dirs['log'])
    trainer = TorchTrainer(env, model, global_counter, summary_writer, in_test, output_path=dirs['data'])
    trainer.run()

    # save model
    final_step = global_counter.cur_step
    logging.info('Training: save final model at step %d ...' % final_step)
    model.save(dirs['model']+'/model.pt', final_step)


def evaluate_fn(agent_dir, output_dir, seeds, port, demo, policy_type):
    agent = agent_dir.split('/')[-1]
    if not check_dir(agent_dir):
        logging.error('Evaluation: %s does not exist!' % agent)
        return
    # load config file for env
    config_dir = find_file(agent_dir + '/data/')
    if not config_dir:
        return
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env, greedy_policy = init_env(config['ENV_CONFIG'], port=port, naive_policy=True)
    logging.info('Evaluation: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r' %
                 (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls))
    env.init_test_seeds(seeds)

    # load model for agent
    if agent != 'greedy':
        # init centralized or multi agent
        # agent名称中包含env类型和reward类型，因此用in来判断
        if 'ippo' in agent:
            model = MultiAgentPolicyPPOWrapper(env.n_s_ls, env.n_a_ls, env.n_w_ls, config['MODEL_CONFIG'])
            print("Eval policy IPPO")
        elif 'ic3netattn' in agent:
            model = MultiAgentPolicyIC3NetAttnWrapper(env.n_s_ls, env.n_a_ls, env.n_w_ls, config['MODEL_CONFIG'])
            print("Eval policy IC3Net with attention")
        elif 'ic3net' in agent:
            model = MultiAgentPolicyIC3NetWrapper(env.n_s_ls, env.n_a_ls, env.n_w_ls, config['MODEL_CONFIG'])
            print("Eval policy IC3Net")
        else:
            logging.error('Evaluation: Not support torch yet')
        if not model.load(agent_dir + '/model/model.pt'):
            return
    else:
        model = greedy_policy
    env.agent = agent
    # collect evaluation data
    evaluator = TorchEvaluator(env, model, output_dir, demo=demo, policy_type=policy_type)
    evaluator.run()


def evaluate(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir, pathes=['eva_data', 'eva_log'])
    init_log(dirs['eva_log'])
    agents = args.agents.split(',')
    # enforce the same evaluation seeds across agents
    seeds = args.evaluation_seeds
    policy_type = args.evaluation_policy_type
    logging.info('Evaluation: policy type: %s, random seeds: %s' % (policy_type, seeds))
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    threads = []
    for i, agent in enumerate(agents):
        agent_dir = base_dir + '/' + agent
        thread = threading.Thread(target=evaluate_fn,
                                  args=(agent_dir, dirs['eva_data'], seeds, i, args.demo, policy_type))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
