"""
A2C, IA2C, MA2C models
@author: Tianshu Chu
"""

import os
from agents.utils import *
from agents.policies import *
import logging
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim


class A2C:
    def __init__(self, n_s, n_a, total_step, model_config, seed=0, n_f=None):
        # load parameters
        self.name = 'a2c'
        self.n_agent = 1
        # init reward norm/clip
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s = n_s
        self.n_a = n_a
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy = self._init_policy(n_s, n_a, n_f, model_config)
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, n_s, n_a, n_w, n_f, model_config, agent_name=None):
        n_fw = model_config.getint('num_fw')
        n_ft = model_config.getint('num_ft')
        n_lstm = model_config.getint('num_lstm')
        if self.name == 'ma2c':
            n_fp = model_config.getint('num_fp')
            policy = FPLstmACPolicy(n_s, n_a, n_w, n_f, self.n_step, n_fc_wave=n_fw,
                                    n_fc_wait=n_ft, n_fc_fp=n_fp, n_lstm=n_lstm, name=agent_name)
        else:
            policy = LstmACPolicy(n_s, n_a, n_w, self.n_step, n_fc_wave=n_fw,
                                  n_fc_wait=n_ft, n_lstm=n_lstm, name=agent_name)
        return policy

    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        beta_init = model_config.getfloat('entropy_coef_init')
        beta_decay = model_config.get('entropy_decay')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('LR_MIN')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)
        if beta_decay == 'constant':
            self.beta_scheduler = Scheduler(beta_init, decay=beta_decay)
        else:
            beta_min = model_config.getfloat('ENTROPY_COEF_MIN')
            beta_ratio = model_config.getfloat('ENTROPY_RATIO')
            self.beta_scheduler = Scheduler(beta_init, beta_min, self.total_step * beta_ratio,
                                            decay=beta_decay)

    def _init_train(self, model_config):
        # init loss
        v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        self.policy.prepare_loss(v_coef, max_grad_norm, alpha, epsilon)

        # init replay buffer
        gamma = model_config.getfloat('gamma')
        self.trans_buffer = OnPolicyBuffer(gamma)

    def save(self, model_dir, global_step):
        self.saver.save(self.sess, model_dir + 'checkpoint', global_step=global_step)

    def load(self, model_dir, checkpoint=None):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-' + str(int(checkpoint))
        if save_file is not None:
            self.saver.restore(self.sess, model_dir + save_file)
            logging.info('Checkpoint loaded: %s' % save_file)
            return True
        logging.error('Can not find old checkpoint for %s' % model_dir)
        return False

    def reset(self):
        self.policy._reset()

    def backward(self, R, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)
        obs, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(R)
        self.policy.backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                             summary_writer=summary_writer, global_step=global_step)

    def forward(self, ob, done, out_type='pv'):
        return self.policy.forward(self.sess, ob, done, out_type)

    def add_transition(self, ob, action, reward, value, done):
        # Hard code the reward norm for negative reward only
        if (self.reward_norm):
            reward /= self.reward_norm
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        self.trans_buffer.add_transition(ob, action, reward, value, done)


class IA2C(A2C):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, total_step,
                 model_config, seed=0):
        self.name = 'ia2c'
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_s, n_w, n_a) in enumerate(zip(self.n_s_ls, self.n_w_ls, self.n_a_ls)):
            # agent_name is needed to differentiate multi-agents
            self.policy_ls.append(self._init_policy(n_s - n_w, n_a, n_w, 0, model_config, agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_train(self, model_config):
        # init loss
        v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        gamma = model_config.getfloat('gamma')
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(v_coef, max_grad_norm, alpha, epsilon)
            self.trans_buffer_ls.append(OnPolicyBuffer(gamma))

    def backward(self, R_ls, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)
        for i in range(self.n_agent):
            obs, acts, dones, Rs, Advs = self.trans_buffer_ls[i].sample_transition(R_ls[i])
            if i == 0:
                self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                                           summary_writer=summary_writer, global_step=global_step)
            else:
                self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta)

    def forward(self, obs, done, out_type='pv'):
        if len(out_type) == 1:
            out = []
        elif len(out_type) == 2:
            out1, out2 = [], []
        for i in range(self.n_agent):
            cur_out = self.policy_ls[i].forward(self.sess, obs[i], done, out_type)
            if len(out_type) == 1:
                out.append(cur_out)
            else:
                out1.append(cur_out[0])
                out2.append(cur_out[1])
        if len(out_type) == 1:
            return out
        else:
            return out1, out2

    def backward_mp(self, R_ls, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)

        def worker(i):
            obs, acts, dones, Rs, Advs = self.trans_buffer_ls[i].sample_transition(R_ls[i])
            self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                                       summary_writer=summary_writer, global_step=global_step)
        mps = []
        for i in range(self.n_agent):
            p = mp.Process(target=worker, args=(i))
            p.start()
            mps.append(p)
        for p in mps:
            p.join()

    def reset(self):
        for policy in self.policy_ls:
            policy._reset()

    def add_transition(self, obs, actions, rewards, values, done):
        if (self.reward_norm):
            rewards = rewards / self.reward_norm
        if self.reward_clip:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i],
                                                   rewards[i], values[i], done)



class MA2C(IA2C):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, n_f_ls, total_step,
                 model_config, seed=0):
        self.name = 'ma2c'
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_f_ls = n_f_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_s, n_a, n_w, n_f) in enumerate(zip(self.n_s_ls, self.n_a_ls, self.n_w_ls, self.n_f_ls)):
            # agent_name is needed to differentiate multi-agents
            self.policy_ls.append(self._init_policy(n_s - n_f - n_w, n_a, n_w, n_f, model_config,
                                                    agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())


class IQL(A2C):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, total_step, model_config, seed=0, model_type='dqn'):
        self.name = 'iql'
        self.model_type = model_type
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_s, n_a, n_w) in enumerate(zip(self.n_s_ls, self.n_a_ls, self.n_w_ls)):
            # agent_name is needed to differentiate multi-agents
            self.policy_ls.append(self._init_policy(n_s, n_a, n_w, model_config,
                                                    agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.cur_step = 0
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, n_s, n_a, n_w, model_config, agent_name=None):
        if self.model_type == 'dqn':
            n_h = model_config.getint('num_h')
            n_fc = model_config.getint('num_fc')
            policy = DeepQPolicy(n_s - n_w, n_a, n_w, self.n_step, n_fc0=n_fc, n_fc=n_h,
                                 name=agent_name)
        else:
            policy = LRQPolicy(n_s, n_a, self.n_step, name=agent_name)
        return policy

    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        eps_init = model_config.getfloat('epsilon_init')
        eps_decay = model_config.get('epsilon_decay')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)
        if eps_decay == 'constant':
            self.eps_scheduler = Scheduler(eps_init, decay=eps_decay)
        else:
            eps_min = model_config.getfloat('epsilon_min')
            eps_ratio = model_config.getfloat('epsilon_ratio')
            self.eps_scheduler = Scheduler(eps_init, eps_min, self.total_step * eps_ratio,
                                           decay=eps_decay)

    def _init_train(self, model_config):
        # init loss
        max_grad_norm = model_config.getfloat('max_grad_norm')
        gamma = model_config.getfloat('gamma')
        buffer_size = model_config.getfloat('buffer_size')
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(max_grad_norm, gamma)
            self.trans_buffer_ls.append(ReplayBuffer(buffer_size, self.n_step))

    def backward(self, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        if self.trans_buffer_ls[0].size < self.trans_buffer_ls[0].batch_size:
            return
        for i in range(self.n_agent):
            for k in range(10):
                obs, acts, next_obs, rs, dones = self.trans_buffer_ls[i].sample_transition()
                if i == 0:
                    self.policy_ls[i].backward(self.sess, obs, acts, next_obs, dones, rs, cur_lr,
                                               summary_writer=summary_writer,
                                               global_step=global_step + k)
                else:
                    self.policy_ls[i].backward(self.sess, obs, acts, next_obs, dones, rs, cur_lr)

    def forward(self, obs, mode='act', stochastic=False):
        if mode == 'explore':
            eps = self.eps_scheduler.get(1)
        action = []
        qs_ls = []
        for i in range(self.n_agent):
            qs = self.policy_ls[i].forward(self.sess, obs[i])
            if (mode == 'explore') and (np.random.random() < eps):
                action.append(np.random.randint(self.n_a_ls[i]))
            else:
                if not stochastic:
                    action.append(np.argmax(qs))
                else:
                    qs = qs / np.sum(qs)
                    action.append(np.random.choice(np.arange(len(qs)), p=qs))
            qs_ls.append(qs)
        return action, qs_ls

    def reset(self):
        # do nothing
        return

    def add_transition(self, obs, actions, rewards, next_obs, done):
        if (self.reward_norm):
            rewards = rewards / self.reward_norm
        if self.reward_clip:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i],
                                                   rewards[i], next_obs[i], done)

# torch model抽象类
class TorchWrapper:
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, model_config):
        self.n_agent = len(n_s_ls)
        self.policy_ls = []
        self.parameters = []
        self.optimizer = None
        self.scheduler = None
        self.n_step = model_config.getint('batch_size')
        # Buffer for reinforcement learning
        self.buffer = PPOBuffer(
            gamma=model_config.getfloat("gamma"),
            lam=model_config.getfloat("gae_lambda", fallback=0.95)
        )

    def add_transition(self, obs, action, reward, value, done):
        self.buffer.add_transition(obs, action, reward, value, done)
        
    def reset(self):
        for policy in self.policy_ls:
            policy.reset()

    def backward(self, R, global_step=None):
        self.optimizer.zero_grad()
        total_loss = self.get_loss(R)
        total_loss.backward()
        # ppo clip
        nn.utils.clip_grad_norm_(self.parameters, max_norm=0.5)
        self.optimizer.step()
        self.scheduler.step()
            
    def save(self, path, final_step=None):
        state_dict = {
            'policies': [policy.state_dict() for policy in self.policy_ls],
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(state_dict, path)
        print(f"Model saved to {path}")

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        for policy, policy_state in zip(self.policy_ls, checkpoint['policies']):
            policy.load_state_dict(policy_state)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"Model loaded from {path}")
        return True

class MultiAgentPolicyPPOWrapper(TorchWrapper):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, model_config):
        super(MultiAgentPolicyPPOWrapper, self).__init__(n_s_ls, n_a_ls, n_w_ls, model_config)
        self.name = 'ippo'

        # 加载类的模型参数，优化器和lr scheduler
        # 加载参数
        for i, (n_s, n_w, n_a) in enumerate(zip(n_s_ls, n_w_ls, n_a_ls)):
            policy = TorchLstmPolicy(
                n_s=n_s - n_w, 
                n_a=n_a, 
                n_w=n_w,
                n_step = self.n_step,
                model_config=model_config,
                agent_name=f"{i}_ppo_agent"
            )
            self.policy_ls.append(policy)
        
        for policy in self.policy_ls:
            self.parameters += list(policy.parameters())
        self.optimizer = optim.Adam(self.parameters , lr=model_config.getfloat('lr_init'))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma= 0.95)

            
    def forward(self, obs, done, out_type='pv'):
        out1, out2 = [], []
        for i in range(self.n_agent):
            cur_out = self.policy_ls[i](obs[i], done, out_type)
            if isinstance(cur_out, tuple):
                out1.append(cur_out[0])
                out2.append(cur_out[1])
            else:
                out1.append(cur_out)
        if out2:
            return out1, out2
        else:
            return out1
    
    def get_loss(self, R):
        obs, actions, dones, returns, advantages = self.buffer.compute_returns_and_advantages(R)
        policy_loss, value_loss, entropy_loss = 0.0, 0.0, 0.0

        for t in range(len(obs)):
            ob = obs[t]
            action = actions[t].detach()
            done = dones[t].detach()
            returnt = returns[t].detach()
            advantage = advantages[t].detach()
            for i, policy in enumerate(self.policy_ls):
                # Actor部分
                policy.reset()
                logits, values = policy.forward(ob[i], done, out_type='pv')
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(action[i])
                entropy = dist.entropy().mean()

                # Advantage
                adv = advantage[i]
                log_probs = log_probs.squeeze()
                policy_loss_i = -(log_probs * adv).mean()

                # Critic部分
                returns_i = returnt[i]
                values = values.squeeze()
                value_loss_i = F.mse_loss(values, returns_i)

                # 累加
                policy_loss += policy_loss_i
                value_loss += value_loss_i
                entropy_loss += entropy
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
        self.buffer.clear()
        return total_loss
        
        
class MultiAgentPolicyIC3NetWrapper(TorchWrapper):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, model_config):
        super(MultiAgentPolicyIC3NetWrapper, self).__init__(n_s_ls, n_a_ls, n_w_ls, model_config)
        self.name = 'ic3net'

        # 加载类的模型参数，优化器和lr scheduler
        self.n_c = model_config.getint('num_lstm', fallback=128)
        for i, (n_s, n_w, n_a) in enumerate(zip(n_s_ls, n_w_ls, n_a_ls)):
            policy = TorchLstmIC3NetPolicy(
                n_s=n_s - n_w, 
                n_a=n_a, 
                n_w=n_w,
                n_step = self.n_step,
                model_config=model_config,
                agent_name=f"{i}a"
            )
            self.policy_ls.append(policy)
        
        for policy in self.policy_ls:
            self.parameters += list(policy.parameters())
        self.optimizer = optim.Adam(self.parameters, lr=model_config.getfloat('lr_init'))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma= 0.95) 
        # commuicate signal
        self.comm = torch.zeros((1, self.n_c))

            
    def forward(self, obs, done, out_type='pv'):
        out1, out2 = [], []
        comm = torch.zeros((1, self.n_c))
        for i in range(self.n_agent):
            cur_out = self.policy_ls[i].forward(obs[i], done, self.comm, out_type)
            if out_type=='pv':
                out1.append(cur_out[0])
                out2.append(cur_out[1])
            else:
                out1.append(cur_out[0])
            comm = comm + cur_out[-1]
        self.comm = comm/self.n_agent
        if out2:
            return out1, out2
        else:
            return out1
        
    def get_loss(self, R):
        obs, actions, dones, returns, advantages = self.buffer.compute_returns_and_advantages(R)
        policy_loss, value_loss, entropy_loss = 0.0, 0.0, 0.0
        self.comm = torch.zeros((1, self.n_c))

        for t in range(len(obs)):
            ob = obs[t]
            action = actions[t].detach()
            done = dones[t].detach()
            returnt = returns[t].detach()
            advantage = advantages[t].detach()
            comm = torch.zeros((1, self.n_c))
            for i, policy in enumerate(self.policy_ls):
                # Actor部分
                policy.reset()
                logits, values, commi = policy.forward(ob[i], done, self.comm, out_type='pv')
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(action[i])
                entropy = dist.entropy().mean()
                comm = comm + commi
                # Advantage
                adv = advantage[i]
                log_probs = log_probs.squeeze()
                policy_loss_i = -(log_probs * adv).mean()

                # Critic部分
                returns_i = returnt[i]
                values = values.squeeze()
                value_loss_i = F.mse_loss(values, returns_i)

                # 累加
                policy_loss += policy_loss_i
                value_loss += value_loss_i
                entropy_loss += entropy
            self.comm = comm / self.n_agent

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
        self.buffer.clear()
        return total_loss

    def reset(self):
        for policy in self.policy_ls:
            policy.reset()
        # 需要额外清空communicate signal
        self.comm = torch.zeros((1, self.n_c))
        
        
class MultiAgentPolicyIC3NetAttnWrapper(TorchWrapper):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, model_config):
        super(MultiAgentPolicyIC3NetAttnWrapper, self).__init__(n_s_ls, n_a_ls, n_w_ls, model_config)
        self.name = 'ic3netattn'
        self.n_c = model_config.getint('num_lstm', fallback=128)

        for i, (n_s, n_w, n_a) in enumerate(zip(n_s_ls, n_w_ls, n_a_ls)):
            policy = TorchLstmIC3NetAttnPolicy(
                n_s=n_s - n_w, 
                n_a=n_a, 
                n_w=n_w,
                n_step = self.n_step,
                model_config=model_config,
                agent_name=f"{i}a"
            )
            self.policy_ls.append(policy)
        
        for policy in self.policy_ls:
            self.parameters += list(policy.parameters())
        self.optimizer = optim.Adam(self.parameters, lr=model_config.getfloat('lr_init'))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma= 0.95)
        self.comm = torch.zeros((self.n_agent, self.n_c))

            
    def forward(self, obs, done, out_type='pv'):
        out1, out2 = [], []
        comm = []
        for i in range(self.n_agent):
            cur_out = self.policy_ls[i].forward(obs[i], done, self.comm, out_type)
            if out_type=='pv':
                out1.append(cur_out[0])
                out2.append(cur_out[1])
            else:
                out1.append(cur_out[0])
            comm.append(cur_out[-1])
        self.comm = torch.stack(comm, dim=0)
        if out2:
            return out1, out2
        else:
            return out1

    def get_loss(self, R):
        obs, actions, dones, returns, advantages = self.buffer.compute_returns_and_advantages(R)
        policy_loss, value_loss, entropy_loss = 0.0, 0.0, 0.0
        self.comm = torch.zeros((self.n_agent, self.n_c))

        for t in range(len(obs)):
            ob = obs[t]
            action = actions[t].detach()
            done = dones[t].detach()
            returnt = returns[t].detach()
            advantage = advantages[t].detach()
            comm = []
            for i, policy in enumerate(self.policy_ls):
                # Actor部分
                policy.reset()
                logits, values, commi = policy.forward(ob[i], done, self.comm, out_type='pv')
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(action[i])
                entropy = dist.entropy().mean()
                comm.append(commi)
                # Advantage
                adv = advantage[i]
                log_probs = log_probs.squeeze()
                policy_loss_i = -(log_probs * adv).mean()

                # Critic部分
                returns_i = returnt[i]
                values = values.squeeze()
                value_loss_i = F.mse_loss(values, returns_i)

                # 累加
                policy_loss += policy_loss_i
                value_loss += value_loss_i
                entropy_loss += entropy
            self.comm = torch.stack(comm, dim=0)

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
        self.buffer.clear()
        return total_loss

    def reset(self):
        for policy in self.policy_ls:
            policy.reset()
        # 需要额外清空communicate signal
        self.comm = torch.zeros((self.n_agent, self.n_c))
