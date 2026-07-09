#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stripped-down baseline matching the recipe in https://github.com/svpino/lunar-lander
as closely as this codebase's infrastructure allows. Used to sanity-check whether
the instability/divergence seen in ModelTrain.py's runs (LL_60/LL_61: QStd climbing
unboundedly, Return never solving) is caused by the extra machinery this project has
stacked on top of a plain DQN (Double DQN, 256x256 net, cosine LR schedule, reward
scaling, gradient clipping, PER) rather than being fundamental to LunarLander itself.

Deviations from ModelTrain.py, and why:
  - Algorithm:     dqn_agent.DqnAgent (plain DQN)     vs SelectiveClipDqnAgent(DdqnAgent) (Double DQN)
  - Network:       1 hidden layer, 32 units, ReLU     vs 2 hidden layers, 256 units each
  - Learning rate: fixed 1e-4, no schedule            vs warmup + cosine decay
  - Batch size:    32                                 vs 256
  - Target update: hard copy (tau=1.0) after every    vs soft update (tau=0.001) every
                    completed episode (done manually     15 train steps (built-in periodic
                    in train(), see below)                updater)
  - Replay:        plain uniform, no PER               vs prioritized (in most other sweeps)
  - Reward scale:  1.0 (raw reward)                     vs 0.1 in LL_61
  - Grad clipping: none                                 vs clipped (global or per-layer)
  - Epsilon decay: multiplicative, once per episode      vs exponential, once per step
                    (cfg.epsilon_decay reinterpreted
                    as a per-episode multiplier here —
                    NOT the same semantics as in ModelTrain.py)

Approximation: tf-agents' built-in target-update mechanism (DqnAgent's
`_get_target_updater`) only supports periodic updates measured in train-step calls,
not environment-episode boundaries. To match the reference's "hard-copy after every
episode" exactly, the built-in updater is disabled (target_update_period set beyond
the run length) and the hard copy is instead performed manually in train() whenever
the collected time_step is terminal.

Episode-based stopping: the reference trains for a fixed 5,000 episodes. The rest of
this codebase's infrastructure (eval_interval, logging cadence, early stopping) is
all expressed in steps, so num_iterations is kept as a generous step budget (safety
cap) and an explicit episode counter is checked against cfg._episode_limit as the
stopping condition that actually mirrors the reference's run length.
"""

import os
import sys
import signal
from datetime import datetime

import math
import copy

import numpy as np

import gymnasium as gym

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout

from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import tensor_spec
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import sequential
from tf_agents.utils import common
from tf_agents.policies import py_tf_eager_policy, random_py_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import py_driver
from tf_agents.networks.layer_utils import print_summary
from tf_agents.utils import eager_utils

from ModelCfg import ModelCfg
import ModelUtils as mutils
from gym_wrap import GymnasiumWrapper


class ModelTrainMin(object):
    """Train model - stripped-down baseline (plain DQN, small net, fixed LR)"""

    #Correctly finish train by CTRL+C
    finish_train = False

    @staticmethod
    def handler(signum, frame):
        """Signal processing handler"""
        signame = signal.Signals(signum).name
        print(f'Signal handler called with signal {signame} ({signum})')
        ModelTrainMin.finish_train = True

    def __init__(self, cfg:ModelCfg) -> None:
        self._mcfg = cfg

        self._train_py_env = gym.make(self._mcfg.env_name)
        self._train_py_env.reset()

        self._eval_py_env = gym.make(self._mcfg.env_name)
        self._eval_py_env.reset()

        self._train_env = GymnasiumWrapper(self._train_py_env)
        self._eval_env = GymnasiumWrapper(self._eval_py_env)
        self._tf_eval_env = tf_py_environment.TFPyEnvironment(self._eval_env)

        self._num_actions = mutils.tensor_size(self._train_env.action_spec())
        self._observations = mutils.tensor_size(self._train_env.time_step_spec().observation)

        self.q_net = None
        self.agent = None
        self.replay_buffer = None
        self.rb_observer = None

        self.ckpt = None
        self.ckpt_manager = None
        self.best_ckpt_manager = None
        self.ckpt_restored=False
        self.tb_writer = None

        # keep-best / early-stop tracking
        self.best_return = float('-inf')
        self.no_improve_evals = 0

        self._debug = False

        self.prev_weights = None
        self.prev_weights_collection = {}

    @property
    def debug(self) -> bool:
        return self._debug
    @debug.setter
    def debug(self, val:bool) -> None:
        self._debug = val

    def initialise(self) -> None:
        self.init_qnet()
        self.init_agent()
        self.init_train_data()
        self.init_checkpoints()
        self.tb_writer = tf.summary.create_file_writer(self._mcfg.tensorboard_dir)


    def collect_episode(self, environment, num_episodes=None, agent=None, num_steps=0, time_step=None) -> any:
        """Collect data for episode"""
        collect_policy = py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True) if agent \
            else random_py_policy.RandomPyPolicy(environment.time_step_spec(), environment.action_spec())

        initial_time_step = time_step if time_step else environment.reset()

        driver = py_driver.PyDriver(
            env=environment,
            policy=collect_policy,
            observers=[self.rb_observer],
            end_episode_on_boundary=True,
            max_steps=num_steps,
            max_episodes=num_episodes)

        last_time_step, policy_state = driver.run(initial_time_step)
        return last_time_step


    def compute_avg_return(self, environment, policy, num_episodes=10):
        total_return = 0.0
        for eps in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0
            steps = 0
            tm_start = datetime.now()

            while not time_step.is_last() and steps < 1000:
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
                steps = steps + 1
            total_return += episode_return

            if self.debug:
                print('Evaluation episode: {0} Rewards: {1:0.2f} {2} steps {3} Duration {4} sec'.format(
                    eps,
                    episode_return.numpy()[0],
                    time_step.reward.numpy(),
                    steps,
                    (datetime.now()-tm_start).seconds)
                    )

        return (total_return / num_episodes).numpy()[0]

    def create_layer(self, idx, lyr_size, lyr_bias, lyr_kernel, lyr_dropout) -> list:
        return [
            Dense(
                lyr_size,
                activation=tf.keras.activations.relu,
                name="LYR_{}".format(idx),
                kernel_initializer=lyr_kernel,
                bias_initializer=lyr_bias
                ),
            Dropout(lyr_dropout)
        ] if lyr_dropout > 0 else [
                Dense(
                    lyr_size,
                    activation=tf.keras.activations.relu,
                    name="LYR_{}".format(idx),
                    kernel_initializer=lyr_kernel,
                    bias_initializer=lyr_bias
                    )
        ]

    def init_qnet(self) -> None:
        layers = []
        for idx in range(len(self._mcfg.layer_sz)):
            layers = layers + self.create_layer(idx, self._mcfg.layer_sz[idx], self._mcfg.bias[idx], self._mcfg.kernel_init[idx], self._mcfg.dropout[idx])

        """
        Output layer - number of units equal number of actions (4 in our case)
        """
        q_values_layer = Dense(
            self._num_actions,
            activation=None,
            name="Output",
            kernel_initializer=self._mcfg.kernel_init_lyr_out,
            bias_initializer=self._mcfg.bias_lyr_out)

        self.q_net = sequential.Sequential(layers + [q_values_layer], input_spec=self._train_env.time_step_spec().observation, name="QNet")

    def init_agent(self) -> None:
        # BASELINE: fixed learning rate, no schedule.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self._mcfg.lrn_rate)

        self.train_step_counter = tf.Variable(0)

        # BASELINE: plain DQN (not Double DQN), no gradient clipping. target_update_tau/
        # period are passed through but the built-in periodic updater is effectively
        # disabled (period set beyond the run length) -- the real target update is the
        # manual per-episode hard copy in train().
        self.agent = dqn_agent.DqnAgent(
                self._train_env.time_step_spec(),
                self._train_env.action_spec(),
                q_network=self.q_net,
                optimizer=self.optimizer,
                target_update_tau=1.0,
                target_update_period=self._mcfg.num_iterations + 1,
                gradient_clipping=None,
                gamma=self._mcfg.gamma,
                reward_scale_factor=self._mcfg.reward_scale_factor,
                epsilon_greedy=self._mcfg.epsilon_start,
                n_step_update=self._mcfg.n_step_update,
                td_errors_loss_fn=common.element_wise_huber_loss,
                train_step_counter=self.train_step_counter)

        self.agent.initialize()
        self.agent.train = common.function(self.agent.train)

    def _as_float_array(self, value):
        value = value.numpy() if hasattr(value, 'numpy') else value
        return np.asarray(value, dtype=np.float64).reshape(-1)

    def _summary_stats(self, value):
        values = self._as_float_array(value)
        if values.size == 0:
            return [float('nan')] * 4
        return [
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.max(values))]

    def _shape_rank(self, shape):
        if hasattr(shape, 'rank'):
            return shape.rank
        return len(shape) if shape is not None else None

    def _log_diagnostics(self, step:int, prefix:str, headers:list, row:list) -> None:
        for name, value in zip(headers[1:], row[1:]):
            if np.isfinite(value):
                mutils.log_scalar(step, "{}/{}".format(prefix, name), value, self.tb_writer)

    def _first_replay_field(self, value):
        value = tf.convert_to_tensor(value)
        if value.shape.rank == 0:
            return value
        value = tf.reshape(value, [tf.shape(value)[0], -1])
        return value[:, 0]

    def _sample_info_values(self, sample_info, name:str):
        if not hasattr(sample_info, name):
            return None
        return self._as_float_array(self._first_replay_field(getattr(sample_info, name)))

    def qvalue_diag_headers(self) -> list:
        return [
            'Step',
            'QMean', 'QStd', 'QMin', 'QMax',
            'MaxQMean', 'MaxQStd', 'MaxQMin', 'MaxQMax',
            'ActionGapMean', 'ActionGapStd', 'ActionGapMin', 'ActionGapMax']

    def replay_diag_headers(self) -> list:
        return [
            'Step', 'Frames',
            'TableSizeMean', 'TableSizeMin', 'TableSizeMax',
            'KeyMin', 'KeyMax', 'KeySpan', 'KeyUnique', 'KeyUniqueFrac',
            'ProbMean', 'ProbStd', 'ProbMin', 'ProbMax',
            'PriorityMean', 'PriorityStd', 'PriorityMin', 'PriorityMax',
            'WeightMean', 'WeightStd', 'WeightMin', 'WeightMax',
            'Beta']

    def collect_qvalue_diagnostics(self, step:int, trajectories) -> list:
        observations = trajectories.observation
        obs_rank = self._shape_rank(observations.shape)
        spec_rank = self._shape_rank(self._train_env.time_step_spec().observation.shape)
        if obs_rank is not None and spec_rank is not None and obs_rank == spec_rank + 2:
            observations = observations[:, 0, ...]

        q_values = self.q_net(observations, training=False)
        if isinstance(q_values, tuple):
            q_values = q_values[0]

        q_arr = np.asarray(q_values.numpy(), dtype=np.float64)
        q_arr = q_arr.reshape(-1, q_arr.shape[-1])
        max_q = np.max(q_arr, axis=1)
        if q_arr.shape[1] > 1:
            top2 = np.sort(q_arr, axis=1)[:, -2:]
            action_gap = top2[:, 1] - top2[:, 0]
        else:
            action_gap = np.zeros_like(max_q)

        row = [float(step)]
        row.extend(self._summary_stats(q_arr))
        row.extend(self._summary_stats(max_q))
        row.extend(self._summary_stats(action_gap))
        self._log_diagnostics(step, 'QValues', self.qvalue_diag_headers(), row)
        return row

    def collect_replay_diagnostics(self, step:int, sample_info, num_frames, weights=None, beta:float=float('nan')) -> list:
        frames = float(num_frames.numpy()) if hasattr(num_frames, 'numpy') else float(num_frames)
        table_size = self._sample_info_values(sample_info, 'table_size')
        keys = self._sample_info_values(sample_info, 'key')
        probs = self._sample_info_values(sample_info, 'probability')
        priorities = self._sample_info_values(sample_info, 'priority')

        table_stats = [float('nan')] * 3
        if table_size is not None and table_size.size > 0:
            table_stats = [float(np.mean(table_size)), float(np.min(table_size)), float(np.max(table_size))]

        key_stats = [float('nan')] * 5
        if keys is not None and keys.size > 0:
            unique = len(np.unique(keys))
            key_stats = [
                float(np.min(keys)),
                float(np.max(keys)),
                float(np.max(keys) - np.min(keys)),
                float(unique),
                float(unique / keys.size)]

        prob_stats = self._summary_stats(probs) if probs is not None else [float('nan')] * 4
        priority_stats = self._summary_stats(priorities) if priorities is not None else [float('nan')] * 4
        weight_stats = self._summary_stats(weights) if weights is not None else [float('nan')] * 4

        row = [float(step), frames]
        row.extend(table_stats)
        row.extend(key_stats)
        row.extend(prob_stats)
        row.extend(priority_stats)
        row.extend(weight_stats)
        row.append(float(beta))
        self._log_diagnostics(step, 'Replay', self.replay_diag_headers(), row)
        return row

    def init_train_data(self) -> None:
        """Prepare replay buffer"""
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=1,
            max_length=self._mcfg.replay_buffer_capacity)

        self.rb_observer = lambda traj: self.replay_buffer.add_batch(
            tf.nest.map_structure(lambda t: tf.expand_dims(tf.constant(t), 0), traj))

    def init_checkpoints(self) -> None:
        if self._mcfg.checkpoint_dir:
            avg_return_var = tf.Variable(0.0, name="compute_avg_return")
            self.ckpt = tf.train.Checkpoint(
                step=tf.Variable(1),
                agent=self.agent,
                policy=self.agent.policy,
                replay_buffer=self.replay_buffer,
                global_step=self.train_step_counter,
                custom_variable=avg_return_var
            )

            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self._mcfg.checkpoint_dir, max_to_keep=self._mcfg.ckpt_max_to_keep)
            # Dedicated "best" checkpoint (keep only 1): overwritten only when a new
            # eval beats the running best, so it survives rotation and late regression.
            best_dir = self._mcfg.checkpoint_dir.rstrip('/') + "_best"
            self.best_ckpt_manager = tf.train.CheckpointManager(self.ckpt, best_dir, max_to_keep=1)
            ckpt_mng_last = self.ckpt_manager.latest_checkpoint

            if self.debug:
                print("Available checkpoints: {}".format(self.ckpt_manager.checkpoints))

            if self._mcfg.if_evaluate_chkpoint:
                # Evaluation mode: do NOT auto-restore the latest checkpoint.
                # evaluate_chkpt() restores the requested checkpoint itself; the
                # auto-restore here would only load the last training attempt
                # (latest ckpt) and print its step/avg, which is misleading.
                print("Eval mode - skipping auto-restore of latest checkpoint")
            elif ckpt_mng_last is not None:
                print("Restore Ckpt from: {}".format(ckpt_mng_last))
                self.ckpt.restore(ckpt_mng_last).expect_partial()
                self.ckpt_restored=True
            else:
                print("No checkpoints")

            if self.debug and self.ckpt_restored:
                print("Loaded checkpoint from: {} Step: {} Save counter: {} {}".format(
                    self._mcfg.checkpoint_dir,
                    self.train_step_counter.numpy(),
                    self.ckpt.save_counter.numpy(),
                    self.ckpt.custom_variable.numpy()))

    def evaluate_chkpt(self, evt_ckpnt:str) -> list:
        """Restore and evaluate checkpoint"""
        if self.debug:
            print("Restore Ckpt from: {}".format(evt_ckpnt))

        self.ckpt.restore(evt_ckpnt).expect_partial()
        avg_return_at_save = float(self.ckpt.custom_variable.numpy())
        step = int(self.train_step_counter.numpy())

        eval_result = []
        for _ in range(3):
            eval_result.append(self.compute_avg_return(self._tf_eval_env, self.agent.policy, self._mcfg.num_eval_episodes))

            if self.finish_train:
                break

        print("Folder: {}/{} Train: Average return: {} Step: {} Avarage for Evaluate: {}".format(
            self._mcfg.checkpoint_dir, self._mcfg.evaluate_chkpoint, avg_return_at_save, step, eval_result))
        return eval_result

    def evaluate(self) -> None:
        """Evaluate all or selected checkpoints"""
        tm_start = datetime.now()
        if self._mcfg.if_evaluate_chkpoint:
            if self._mcfg.evaluate_chkpoint == "all":
                for evt_ckpnt in self.ckpt_manager.checkpoints:
                    if self.debug:
                        print(evt_ckpnt)
                    self.evaluate_chkpt(evt_ckpnt)

                    if self.finish_train:
                        break
            else:
                evt_ckpnt = "{}/ckpt-{}".format(self._mcfg.checkpoint_dir, self._mcfg.evaluate_chkpoint)

                if self.debug:
                    print(evt_ckpnt)
                    print("Available checkpoints: {}".format(self.ckpt_manager.checkpoints))

                if evt_ckpnt not in self.ckpt_manager.checkpoints:
                    evt_ckpnt = self.ckpt_manager.latest_checkpoint

                if evt_ckpnt is None:
                    print("No available checkpoints. Stop eveluation.")
                else:
                    self.evaluate_chkpt(evt_ckpnt)

        print("Evaluation finished..... {}".format(datetime.now() - tm_start))

    def train(self) -> None:

        #Set CTRL+C handler
        signal.signal(signal.SIGINT, ModelTrainMin.handler)

        if self._mcfg.if_evaluate_chkpoint:
            self.evaluate()
            return

        print("Start training.....")

        if self.debug:
            print_summary(self.q_net)

        train_collect_policy = py_tf_eager_policy.PyTFEagerPolicy(self.agent.collect_policy, use_tf_function=True)
        train_driver = py_driver.PyDriver(
            env=self._train_env,
            policy=train_collect_policy,
            observers=[self.rb_observer],
            end_episode_on_boundary=True,
            max_steps=self._mcfg.train_driver_max_step,
            max_episodes=0)

        policy_state = train_collect_policy.get_initial_state(self._train_env.batch_size)
        print(self._train_env.batch_size)

        #put initial number of records to buffer
        train_time_step = self.collect_episode(self._train_env,
                                               num_steps=self._mcfg.num_initial_records)
        f_step = self.agent.train_step_counter.numpy()
        returns = mutils.read_results(self._mcfg.results_file)

        if self.debug:
            print("Frames in reply buffer: {} First step: {}".format(self.replay_buffer.num_frames(), f_step))
            print(returns)

        avg_return = self.compute_avg_return(self._tf_eval_env, self.agent.policy, self._mcfg.num_eval_episodes)
        returns.append(avg_return)
        mutils.log_scalar(f_step, "Return/eval", avg_return, self.tb_writer)

        tm_start = datetime.now()

        loss_list = []
        grads = []
        qvalue_diag = []
        replay_diag = []

        loss_counter = 0.0

        self.best_return = float('-inf')
        self.no_improve_evals = 0

        # BASELINE: episode-scoped state -- hard target update and epsilon decay both
        # fire once per completed episode, not once per train step.
        self.episode_count = 0
        self._epsilon = self._mcfg.epsilon_start

        mutils.param_gradients(0, self.q_net, grads)
        mutils.log_weight_histograms(0, self.q_net, self.tb_writer)

        iterator = iter(self.replay_buffer.as_dataset(sample_batch_size=self._mcfg.batch_size, num_steps=self._mcfg.sequence_length))

        for _ in range(self._mcfg.num_iterations):
            train_time_step, policy_state = train_driver.run(
                time_step=train_time_step,
                policy_state=policy_state,
            )

            if bool(train_time_step.is_last()):
                self.episode_count += 1

                # BASELINE: hard-copy Q-network weights onto the target network after
                # every completed episode (tau=1.0), matching the reference recipe.
                # The agent's own built-in periodic updater is disabled (see
                # init_agent) so this is the only place the target network changes.
                common.soft_variables_update(
                    self.agent._q_network.variables,
                    self.agent._target_q_network.variables,
                    tau=1.0,
                    tau_non_trainable=1.0,
                )

                # BASELINE: multiplicative epsilon decay once per episode (reference:
                # epsilon *= 0.99941 after each episode), floored at epsilon_end.
                # Note cfg.epsilon_decay is reinterpreted here as that per-episode
                # multiplier -- NOT the per-step exponential rate ModelTrain.py uses.
                self._epsilon = max(self._mcfg.epsilon_end, self._epsilon * self._mcfg.epsilon_decay)
                self.agent.collect_policy._epsilon = self._epsilon

                episode_limit = getattr(self._mcfg, '_episode_limit', None)
                if episode_limit is not None and self.episode_count >= episode_limit:
                    print("Episode limit reached: {} episodes".format(self.episode_count))
                    self.finish_train = True

            num_frames = self.replay_buffer.num_frames()

            # Use data from the buffer and update the agent's network.
            trajectories, sample_info = next(iterator)
            train_loss = self.agent.train(experience=trajectories)
            loss_counter += train_loss.loss

            reward_per_batch = (np.sum(trajectories.reward.numpy())/self._mcfg.batch_size)

            step = self.agent.train_step_counter.numpy()

            if step > 0 and step % self._mcfg.log_interval == 0 and self.debug:
                print('step = {0} ep = {1}: loss = {2:0.5f} Reward: {3:0.3f} ε={4:.4f} Sec. {5} Frames: {6}'.format(
                        step, self.episode_count, loss_counter/self._mcfg.log_interval, reward_per_batch,
                        self._epsilon, (datetime.now()-tm_start).seconds, num_frames))

            if step > 0 and step % self._mcfg.log_loss_interval == 0:
                avg_loss = loss_counter/self._mcfg.log_loss_interval
                loss_list.append([step, avg_loss])
                loss_counter = 0.0

                mutils.log_scalar(step, "Loss/train", avg_loss, self.tb_writer)
                # plain DqnAgent has no _grad_norm_vars (that's SelectiveClipDqnAgent-only);
                # falls back to weight norms, which is fine since this baseline doesn't clip.
                mutils.param_gradients(step, self.q_net, grads)
                mutils.log_weight_histograms(step, self.q_net, self.tb_writer)
                qvalue_diag.append(self.collect_qvalue_diagnostics(int(step), trajectories))
                replay_diag.append(self.collect_replay_diagnostics(int(step), sample_info, num_frames))

            if step > 0 and step % self._mcfg.eval_interval == 0:
                avg_return = self.compute_avg_return(self._tf_eval_env, self.agent.policy, self._mcfg.num_eval_episodes)
                returns.append(avg_return)
                mutils.log_scalar(step, "Return/eval", avg_return, self.tb_writer)
                if self.debug:
                    print('---> Step = {0} Ep = {1}: Average Return = {2:0.2f} All: {3}'.format(step, self.episode_count, avg_return, returns))

                if avg_return > 0 and self.ckpt:
                    self.ckpt.step.assign_add(1)
                    self.ckpt.custom_variable.assign(avg_return)
                    sv_folder = self.ckpt_manager.save()
                    if self.debug:
                        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), sv_folder))

                # Keep-best: a meaningful new high overwrites the single best checkpoint
                # and resets the patience counter; otherwise count an eval without progress.
                if self.ckpt and avg_return >= self.best_return + self._mcfg.early_stop_min_delta:
                    self.best_return = avg_return
                    self.no_improve_evals = 0
                    self.ckpt.custom_variable.assign(avg_return)
                    best_folder = self.best_ckpt_manager.save()
                    if self.debug:
                        print("New best return {:0.2f} at step {} -> {}".format(avg_return, step, best_folder))
                else:
                    self.no_improve_evals += 1

                # Early stopping: stop once solved, or after patience evals without progress.
                if self._mcfg.early_stop_enabled:
                    if avg_return >= self._mcfg.early_stop_target:
                        print("Early stop: solved (avg return {:0.2f} >= {}) at step {}".format(
                            avg_return, self._mcfg.early_stop_target, step))
                        self.finish_train = True
                    elif self.no_improve_evals >= self._mcfg.early_stop_patience:
                        print("Early stop: no improvement for {} evals (best {:0.2f}) at step {}".format(
                            self.no_improve_evals, self.best_return, step))
                        self.finish_train = True

            if self.finish_train:
                break

        avg_return = self.compute_avg_return(self._tf_eval_env, self.agent.policy, self._mcfg.num_eval_episodes)
        returns.append(avg_return)
        mutils.log_scalar(step, "Return/eval", avg_return, self.tb_writer)
        if self.debug:
            print('---> Step = {0} Ep = {1}: Average Return = {2:0.2f} All: {3}'.format(step, self.episode_count, avg_return, returns))

        # final eval may itself be a new best
        if self.ckpt and avg_return >= self.best_return + self._mcfg.early_stop_min_delta:
            self.best_return = avg_return
            self.ckpt.custom_variable.assign(avg_return)
            self.best_ckpt_manager.save()

        if self.best_ckpt_manager and self.best_ckpt_manager.latest_checkpoint:
            print("Best checkpoint: {} (avg return {:0.2f})".format(
                self.best_ckpt_manager.latest_checkpoint, self.best_return))

        if self.ckpt:
            self.ckpt.step.assign_add(1)
            self.ckpt.custom_variable.assign(avg_return)
            sv_folder = self.ckpt_manager.save()
            if self.debug:
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), sv_folder))

        mutils.save_results(self._mcfg.results_file, returns)
        mutils.save_info2cvs(self._mcfg.loss_file, loss_list, ["Step", "Loss"])

        prm_headrs = mutils.param_names(self.q_net)
        mutils.save_info2cvs(self._mcfg.gradient_file, grads, prm_headrs)
        mutils.save_info2cvs(self._mcfg.qvalue_file, qvalue_diag, self.qvalue_diag_headers(), sformat="{:.6f}")
        mutils.save_info2cvs(self._mcfg.replay_diag_file, replay_diag, self.replay_diag_headers(), sformat="{:.6f}")

        mutils.save_info2list(self._mcfg.all_results_file, returns, name=self._mcfg.data_idx)

        mutils.save_parameters(tm_start, self._mcfg.data_idx, [self._mcfg.num_iterations, self._mcfg.batch_size, 1.0,
                        1, self._mcfg.lrn_rate, self._mcfg.gamma,
                        self._mcfg.epsilon_start, self._mcfg.epsilon_end, self._mcfg.epsilon_decay,
                        self._mcfg.gradient_clipping, self._mcfg.num_initial_records, self._mcfg.kernel_init_type],
                        self._mcfg.layer_sz,
                        self._mcfg.clip_layer_names)

        print("Training finished..... {} episodes, {} steps".format(self.episode_count, step))


def _suppress_pool_del_oserror(unraisable):
    import errno
    if (unraisable.exc_type is OSError and
            unraisable.exc_value.errno == errno.EBADF and
            'Pool' in type(unraisable.object).__qualname__):
        return
    sys.__unraisablehook__(unraisable)

if __name__ == '__main__':
    sys.unraisablehook = _suppress_pool_del_oserror

    cfg = ModelCfg()

    label = None
    step_idx = None

    for cmd in sys.argv:
        if cmd.find("--step=") >= 0:
            step_idx = cmd.split('=')[1]
            if len(step_idx)==0:
                print("No folder for evaluation")
                exit()

        if cmd.find("--label=") >= 0:
            label = cmd.split('=')[1]

    cfg.data_idx = label if label else "LL_min_1"

    if step_idx:
        print("Evaluate parameters: {} {}".format(cfg.data_idx, step_idx))
        cfg.evaluate_chkpoint = step_idx

        mdl = ModelTrainMin(cfg=cfg)
        mdl.debug = True
        mdl.initialise()
        mdl.evaluate()
        exit()

    # Baseline recipe matching https://github.com/svpino/lunar-lander as closely as
    # this codebase's infrastructure allows. See module docstring for the full list
    # of deviations from ModelTrain.py and why each one was picked.
    cfg.replay_sampler = 'uniform'              # no PER

    cfg.layer_sz = [32]                         # 1 hidden layer, 32 units (was [256, 256])
    cfg.kernel_init_type = 'GlorotUniform'      # reference doesn't specify; matches plain Keras Dense default
    cfg._kernel_init_lyr_out = tf.keras.initializers.GlorotUniform()  # was a tuned RandomUniform(-0.03, 0.03)
    cfg._bias_lyr_out = tf.keras.initializers.Zeros()                 # was Constant(0) -- same value, explicit default

    cfg._lrn_rate = 0.0001                      # fixed 1e-4, no schedule (was 2.5e-5/1.5e-5/5e-5 cosine)
    cfg._dynamic_lrn_rate = False

    cfg._gamma = 0.99                           # matches ModelCfg default; explicit for clarity

    cfg._batch_size = 32                        # was 256

    cfg.num_iterations = 1500000                # generous step-budget safety cap; episode_limit below is
                                                 # what actually mirrors the reference's run length
    cfg._replay_buffer_capacity = 250000        # keep AFTER num_iterations -- its setter resets this
    cfg._num_initial_records = 1000             # reference doesn't specify a prefill; small warm-up so
                                                 # training can start (batch_size=32 fills fast)

    cfg._epsilon_start = 1.0
    cfg._epsilon_end = 0.01                     # not specified by reference; small floor vs fully greedy
    cfg._epsilon_decay = 0.99941                # NOTE: reinterpreted in train() as a per-episode multiplier

    cfg._reward_scale_factor = 1.0              # no reward scaling (was 0.1 in LL_61)
    cfg._gradient_clipping = None                # no clipping
    cfg._clip_layer_names = []

    cfg._target_update_tau = 1.0                # hard copy -- see init_agent() for how this is actually applied
    cfg._target_update_period = 1

    cfg._episode_limit = 5000                   # mirrors the reference's "5,000 training episodes"

    cfg._num_eval_episodes = 30
    cfg._early_stop_enabled = True
    cfg._early_stop_patience = 15
    cfg._early_stop_min_delta = 5.0
    cfg._early_stop_target = 200.0              # solved threshold, same definition as reference

    mdl = ModelTrainMin(cfg=cfg)
    mdl.debug = True
    mdl.initialise()
    mdl.train()
