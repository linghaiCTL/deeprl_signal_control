import numpy as np
import tensorflow as tf
from agents.utils import *
import bisect
import torch
import torch.nn as nn
import torch.nn.functional as F


class ACPolicy:
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name):
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _build_out_net(self, h, out_type):
        if out_type == 'pi':
            pi = fc(h, out_type, self.n_a, act=tf.nn.softmax)
            return tf.squeeze(pi)
        else:
            v = fc(h, out_type, 1, act=lambda x: x)
            return tf.squeeze(v)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi)
        if 'v' in out_type:
            outs.append(self.v)
        return outs

    def _return_forward_outs(self, out_values):
        if len(out_values) == 1:
            return out_values[0]
        return out_values

    def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.entropy_coef = tf.placeholder(tf.float32, [])
        A_sparse = tf.one_hot(self.A, self.n_a)
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * A_sparse, axis=1) * self.ADV)
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                   epsilon=epsilon)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        # monitor training
        if self.name.endswith('_0a'):
            summaries = []
            # summaries.append(tf.summary.scalar('loss/%s_entropy_loss' % self.name, entropy_loss))
            summaries.append(tf.summary.scalar('loss/%s_policy_loss' % self.name, policy_loss))
            summaries.append(tf.summary.scalar('loss/%s_value_loss' % self.name, value_loss))
            summaries.append(tf.summary.scalar('loss/%s_total_loss' % self.name, self.loss))
            # summaries.append(tf.summary.scalar('train/%s_lr' % self.name, self.lr))
            # summaries.append(tf.summary.scalar('train/%s_entropy_beta' % self.name, self.entropy_coef))
            summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
            self.summary = tf.summary.merge(summaries)


class LstmACPolicy(ACPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, n_fc_wave=128, n_fc_wait=32, n_lstm=64, name=None):
        super().__init__(n_a, n_s, n_step, 'lstm', name)
        self.n_lstm = n_lstm
        self.n_fc_wait = n_fc_wait
        self.n_fc_wave = n_fc_wave
        self.n_w = n_w
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s + n_w]) # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s + n_w]) # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net('forward', 'pi')
            self.v_fw, v_state = self._build_net('forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net('backward', 'pi')
            self.v, _ = self._build_net('backward', 'v')
        self._reset()

    def _build_net(self, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        if self.n_w == 0:
            h = fc(ob, out_type + '_fcw', self.n_fc_wave)
        else:
            h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
            h1 = fc(ob[:, self.n_s:], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1], 1)
        h, new_states = lstm(h, done, states, out_type + '_lstm')
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
        self.states_bw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        # update state only when p is called
        if 'p' in out_type:
            outs.append(self.new_states)
        out_values = sess.run(outs, {self.ob_fw:np.array([ob]),
                                     self.done_fw:np.array([done]),
                                     self.states:self.states_fw})
        if 'p' in out_type:
            self.states_fw = out_values[-1]
            out_values = out_values[:-1]
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                        {self.ob_bw: obs,
                         self.done_bw: dones,
                         self.states: self.states_bw,
                         self.A: acts,
                         self.ADV: Advs,
                         self.R: Rs,
                         self.lr: cur_lr,
                         self.entropy_coef: cur_beta})
        self.states_bw = np.copy(self.states_fw)
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi_fw)
        if 'v' in out_type:
            outs.append(self.v_fw)
        return outs

class TorchLstmPolicy(nn.Module):
    def __init__(self, n_s, n_a, n_w, n_step, model_config, agent_name=None):
        super(TorchLstmPolicy, self).__init__()
        self.name = f"{agent_name}_torch_agent" if agent_name else "torch_agent"
        self.n_s = n_s
        self.n_a = n_a
        self.n_w = n_w
        self.n_step = n_step

        n_fc_wave = model_config.getint('num_fw', fallback=64)
        n_fc_wait = model_config.getint('num_ft', fallback=64)
        n_lstm = model_config.getint('num_lstm', fallback=128)

        # 用torch形式复现网络参数
        self.fc1 = nn.Linear(n_s + n_w, n_fc_wave)
        self.fc2 = nn.Linear(n_fc_wave, n_fc_wait)
        self.lstm = nn.LSTM(n_fc_wait, n_lstm, batch_first=True)
        self.actor_out = nn.Linear(n_lstm, n_a)
        self.critic_out = nn.Linear(n_lstm, 1)
        self.hidden = None

    def forward(self, obs, done, out_type='pv'):
        # obs in shape [n_s]
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        # 初始化lstm
        if done or self.hidden is None:
            self.hidden = (torch.zeros(1, obs.shape[0], self.lstm.hidden_size),
                           torch.zeros(1, obs.shape[0], self.lstm.hidden_size))

        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(1)  # [1, 1, D]
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # Actor
        logits = self.actor_out(lstm_out.squeeze(1).squeeze(0))  # [1, n_a]
        probs = F.softmax(logits, dim=-1)
        # Critic
        value = self.critic_out(lstm_out.squeeze(1).squeeze(0)).squeeze(-1)  # [1]

        if out_type == 'pv':
            return probs, value
        elif out_type == 'p':
            return probs
        elif out_type == 'v':
            return value
        else:
            raise ValueError(f"Unsupported out_type: {out_type}")

    def reset(self):
        self.hidden = None


class TorchLstmIC3NetPolicy(TorchLstmPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, model_config, agent_name=None):
        super(TorchLstmIC3NetPolicy, self).__init__(n_s, n_a, n_w, n_step, model_config, agent_name)
        self.name = f"{agent_name}_ic3net" if agent_name else "ic3net"
        n_lstm = model_config.getint('num_lstm', fallback=128)
        self.n_c = n_lstm # 沟通信号的变量维度和lstm隐变量维度一致
        n_fc_wave = model_config.getint('num_fw', fallback=64)

        # 重构输入层以接收沟通信号
        self.fc1 = nn.Linear(n_s + n_w + self.n_c, n_fc_wave)
        
        # communicate门控网络
        self.comm_gate = nn.Linear(n_lstm, n_lstm)

    def forward(self, obs, done, comm, out_type='pv'):
        # obs in shape [n_s]
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = torch.cat([obs, comm], dim=-1)
        # 重置LSTM
        if done or self.hidden is None:
            self.hidden = (torch.zeros(1, obs.shape[0], self.lstm.hidden_size),
                           torch.zeros(1, obs.shape[0], self.lstm.hidden_size))

        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(1)  # [B, 1, D]
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # Actor
        logits = self.actor_out(lstm_out.squeeze(1).squeeze(0))  # [B, n_a]
        probs = F.softmax(logits, dim=-1)
        # Critic
        value = self.critic_out(lstm_out.squeeze(1).squeeze(0)).squeeze(-1)  # [B]
        # Comm
        comm = F.sigmoid(self.comm_gate(lstm_out)) * lstm_out 
        comm = comm.squeeze(0)

        # 需要额外返回沟通信号，在model类中管理
        if out_type == 'pv':
            return probs, value, comm
        elif out_type == 'p':
            return probs, comm
        elif out_type == 'v':
            return value, comm
        else:
            raise ValueError(f"Unsupported out_type: {out_type}")

class TorchLstmIC3NetAttnPolicy(TorchLstmIC3NetPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, model_config, agent_name=None):
        super(TorchLstmIC3NetAttnPolicy, self).__init__(n_s, n_a, n_w, n_step, model_config, agent_name)
        # 无需父类门控网络
        self.comm_gate = None
        self.name = f"{agent_name}_ic3netattn" if agent_name else "ic3netattn"
        n_lstm = model_config.getint('num_lstm', fallback=128)
        self.n_lstm=n_lstm
        
        # communicate网络，使用attention机制接收输入
        self.comm_q = nn.Linear(n_lstm, n_lstm)
        self.comm_k = nn.Linear(n_lstm, n_lstm)
        self.comm_v = nn.Linear(n_lstm, n_lstm)
        self.comm_out = nn.Linear(n_lstm, n_lstm)
        
    def forward(self, obs, done, comm, out_type='pv'):
        # 基于attention机制处理comm
        comm = comm.unsqueeze(0)  # (1, n_agent, n_lstm)
        Q = self.comm_q(comm)  # (1, n_agent, n_lstm)
        K = self.comm_k(comm)  # (1, n_agent, n_lstm)
        V = self.comm_v(comm)  # (1, n_agent, n_lstm)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.n_lstm ** 0.5)  # (1, n_agent, n_agent)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (1, n_agent, n_agent)
        attn_output = torch.matmul(attn_weights, V)  # (1, n_agent, n_lstm)
        new_comm = self.comm_out(attn_output.mean(dim=1))  # (1, n_lstm)
        
        # obs in shape [n_s]
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = torch.cat([obs, new_comm], dim=-1)
        # 重置LSTM
        if done or self.hidden is None:
            self.hidden = (torch.zeros(1, obs.shape[0], self.lstm.hidden_size),
                           torch.zeros(1, obs.shape[0], self.lstm.hidden_size))

        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(1)  # [1, 1, D]
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # Actor
        logits = self.actor_out(lstm_out.squeeze(1).squeeze(0))  # [1, n_a]
        probs = F.softmax(logits, dim=-1)
        # Critic
        value = self.critic_out(lstm_out.squeeze(1).squeeze(0)).squeeze(-1)  # [1]
        # Comm
        comm = lstm_out 
        comm = comm.squeeze(0).squeeze(0)

        if out_type == 'pv':
            return probs, value, comm
        elif out_type == 'p':
            return probs, comm
        elif out_type == 'v':
            return value, comm
        else:
            raise ValueError(f"Unsupported out_type: {out_type}")

class FPLstmACPolicy(LstmACPolicy):
    def __init__(self, n_s, n_a, n_w, n_f, n_step, n_fc_wave=128, n_fc_wait=32, n_fc_fp=32, n_lstm=64, name=None):
        ACPolicy.__init__(self, n_a, n_s, n_step, 'fplstm', name)
        self.n_lstm = n_lstm
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc_fp = n_fc_fp
        self.n_w = n_w
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s + n_w + n_f]) # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s + n_w + n_f]) # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net('forward', 'pi')
            self.v_fw, v_state = self._build_net('forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net('backward', 'pi')
            self.v, _ = self._build_net('backward', 'v')
        self._reset()

    def _build_net(self, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
        h1 = fc(ob[:, (self.n_s + self.n_w):], out_type + '_fcf', self.n_fc_fp)
        if self.n_w == 0:
            h = tf.concat([h0, h1], 1)
        else:
            h2 = fc(ob[:, self.n_s: (self.n_s + self.n_w)], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1, h2], 1)
        h, new_states = lstm(h, done, states, out_type + '_lstm')
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states


class FcACPolicy(ACPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, n_fc_wave=128, n_fc_wait=32, n_lstm=64, name=None):
        super().__init__(n_a, n_s, n_step, 'fc', name)
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc = n_lstm
        self.n_w = n_w
        self.obs = tf.placeholder(tf.float32, [None, n_s + n_w])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi = self._build_net('pi')
            self.v = self._build_net('v')

    def _build_net(self, out_type):
        if self.n_w == 0:
            h = fc(self.obs, out_type + '_fcw', self.n_fc_wave)
        else:
            h0 = fc(self.obs[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
            h1 = fc(self.obs[:, self.n_s:], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1], 1)
        h = fc(h, out_type + '_fc', self.n_fc)
        return self._build_out_net(h, out_type)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        out_values = sess.run(outs, {self.obs: np.array([ob])})
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                        {self.obs: obs,
                         self.A: acts,
                         self.ADV: Advs,
                         self.R: Rs,
                         self.lr: cur_lr,
                         self.entropy_coef: cur_beta})
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)


class FPFcACPolicy(FcACPolicy):
    def __init__(self, n_s, n_a, n_w, n_f, n_step, n_fc_wave=128, n_fc_wait=32, n_fc_fp=32, n_lstm=64, name=None):
        ACPolicy.__init__(self, n_a, n_s, n_step, 'fpfc', name)
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc_fp = n_fc_fp
        self.n_fc = n_lstm
        self.n_w = n_w
        self.obs = tf.placeholder(tf.float32, [None, n_s + n_w + n_f])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi = self._build_net('pi')
            self.v = self._build_net('v')

    def _build_net(self, out_type):
        h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
        h1 = fc(ob[:, (self.n_s + self.n_w):], out_type + '_fcf', self.n_fc_fp)
        if self.n_w == 0:
            h = tf.concat([h0, h1], 1)
        else:
            h2 = fc(ob[:, self.n_s: (self.n_s + self.n_w)], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1, h2], 1)
        h = fc(h, out_type + '_fc', self.n_fc)
        return self._build_out_net(h, out_type)


class QPolicy:
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name):
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _build_fc_net(self, h, n_fc_ls):
        for i, n_fc in enumerate(n_fc_ls):
            h = fc(h, 'q_fc_%d' % i, n_fc)
        q = fc(h, 'q', self.n_a, act=lambda x: x)
        return tf.squeeze(q)

    def _build_net(self):
        raise NotImplementedError()

    def prepare_loss(self, max_grad_norm, gamma):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.S1 = tf.placeholder(tf.float32, [self.n_step, self.n_s + self.n_w])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.DONE = tf.placeholder(tf.bool, [self.n_step])
        A_sparse = tf.one_hot(self.A, self.n_a)

        # backward
        with tf.variable_scope(self.name + '_q', reuse=True):
            q0s = self._build_net(self.S)
            q0 = tf.reduce_sum(q0s * A_sparse, axis=1)
        with tf.variable_scope(self.name + '_q', reuse=True):
            q1s = self._build_net(self.S1)
            q1 = tf.reduce_max(q1s, axis=1)
        tq = tf.stop_gradient(tf.where(self.DONE, self.R, self.R + gamma * q1))
        self.loss = tf.reduce_mean(tf.square(q0 - tq))

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        # monitor training
        if self.name.endswith('_0a'):
            summaries = []
            summaries.append(tf.summary.scalar('train/%s_loss' % self.name, self.loss))
            summaries.append(tf.summary.scalar('train/%s_q' % self.name, tf.reduce_mean(q0)))
            summaries.append(tf.summary.scalar('train/%s_tq' % self.name, tf.reduce_mean(tq)))
            summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
            self.summary = tf.summary.merge(summaries)


class DeepQPolicy(QPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, n_fc0=128, n_fc=64, name=None):
        super().__init__(n_a, n_s, n_step, 'dqn', name)
        self.n_fc = n_fc
        self.n_fc0 = n_fc0
        self.n_w = n_w
        self.S = tf.placeholder(tf.float32, [None, n_s + n_w])
        with tf.variable_scope(self.name + '_q'):
            self.qvalues = self._build_net(self.S)

    def _build_net(self, S):
        if self.n_w == 0:
            h = fc(S, 'q_fcw', self.n_fc0)
        else:
            h0 = fc(S[:, :self.n_s], 'q_fcw', self.n_fc0)
            h1 = fc(S[:, self.n_s:], 'q_fct', self.n_fc0 / 4)
            h = tf.concat([h0, h1], 1)
        return self._build_fc_net(h, [self.n_fc])

    def forward(self, sess, ob):
        return sess.run(self.qvalues, {self.S: np.array([ob])})

    def backward(self, sess, obs, acts, next_obs, dones, rs, cur_lr,
                 summary_writer=None, global_step=None):
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                        {self.S: obs,
                         self.A: acts,
                         self.S1: next_obs,
                         self.DONE: dones,
                         self.R: rs,
                         self.lr: cur_lr})
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)


class LRQPolicy(DeepQPolicy):
    def __init__(self, n_s, n_a, n_step, name=None):
        QPolicy.__init__(self, n_a, n_s, n_step, 'lr', name)
        self.S = tf.placeholder(tf.float32, [None, n_s])
        self.n_w = 0
        with tf.variable_scope(self.name + '_q'):
            self.qvalues = self._build_net(self.S)

    def _build_net(self, S):
        return self._build_fc_net(S, [])
