#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

class SelectiveClipDqnAgent(dqn_agent.DdqnAgent):
#class SelectiveClipDqnAgent(dqn_agent.DqnAgent):
    """DQN agent that applies clipnorm only to selected layers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._grad_norm_vars = None  # storage slot
        self._clip_layer_names = []
        self._clip_norm_value = 0.0

    @property
    def clip_layer_names(self) -> list:
        return self._clip_layer_names

    @clip_layer_names.setter
    def clip_layer_names(self, lnames:list) -> None:
        self._clip_layer_names = lnames

    @property
    def clip_norm_value(self) -> float:
        return self._clip_norm_value

    @clip_norm_value.setter
    def clip_norm_value(self, ln_val:float) -> None:
        self._clip_norm_value = ln_val


    def _ensure_grad_vars(self, gradients):
        """Create tf.Variables to hold gradient norms, once shapes are known."""
        if self._grad_norm_vars is None:
            self._grad_norm_vars = [
                tf.Variable(0.0, trainable=False, dtype=tf.float32, name=f"grad_norm_{i}") for i in range(len(gradients))
            ]

    def _train(self, experience, weights=None):
        with tf.GradientTape() as tape:
            loss_info = self._loss(
                experience,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True,
            )

        variables = self._q_network.trainable_variables
        gradients = tape.gradient(loss_info.loss, variables)

        if self._gradient_clipping is not None:
            grads_and_vars = list(zip(gradients, variables))
            grads_and_vars = eager_utils.clip_gradient_norms(
                grads_and_vars, self._gradient_clipping
            )
            clipped_gradients = [grad for grad, _ in grads_and_vars]
        else:
            clipped_gradients = []
            for grad, var in zip(gradients, variables):
                if grad is None:
                    clipped_gradients.append(grad)
                elif any(lyr in var.name for lyr in self.clip_layer_names):
                    clipped_gradients.append(tf.clip_by_norm(grad, self.clip_norm_value))
                else:
                    clipped_gradients.append(grad)

        self._optimizer.apply_gradients(zip(clipped_gradients, variables))

        # Ensure storage variables exist (only creates them once)
        self._ensure_grad_vars(clipped_gradients)

        # assign() is a graph op — runs on EVERY call, not just trace time
        for i, grad in enumerate(clipped_gradients):
            if grad is not None:
                self._grad_norm_vars[i].assign(tf.norm(grad))
            else:
                self._grad_norm_vars[i].assign(0.0)

        self.train_step_counter.assign_add(1)
        self._update_target()
        return loss_info


class ModelTrain(object):
    """Train model"""

    #Correctly finish train by CTRL+C
    finish_train = False

    @staticmethod
    def handler(signum, frame):
        """Signal processing handler"""
        signame = signal.Signals(signum).name
        print(f'Signal handler called with signal {signame} ({signum})')
        ModelTrain.finish_train = True

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
        self.best_return_ckpt_var = None  # set in setup(); persisted in checkpoint

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
        self.optimizer = None
        if self._mcfg.dynamic_lrn_rate:
            decay_steps = self._mcfg.cosine_decay_steps if self._mcfg.cosine_decay_steps else self._mcfg.num_iterations
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self._mcfg.lrn_rate*0.1,
                decay_steps=decay_steps,
                alpha=self._mcfg.cosine_decay_alpha,     # floor = alpha * peak; LR holds at floor for remainder of run
                warmup_target=self._mcfg.lrn_rate,
                warmup_steps=decay_steps * 0.1
            )
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self._mcfg.lrn_rate)

        self.train_step_counter = tf.Variable(0)
        self.agent = SelectiveClipDqnAgent(
                self._train_env.time_step_spec(),
                self._train_env.action_spec(),
                q_network=self.q_net,
                optimizer=self.optimizer,
                target_update_tau=self._mcfg.target_update_tau,
                target_update_period=self._mcfg.target_update_period,
                gradient_clipping=self._mcfg.gradient_clipping if len(self._mcfg.clip_layer_names)==0 else None, #gradient_clipping,
                gamma=self._mcfg.gamma,
                reward_scale_factor=self._mcfg.reward_scale_factor,
                epsilon_greedy=self._mcfg.epsilon_start,
                n_step_update=self._mcfg.n_step_update,
                td_errors_loss_fn=common.element_wise_huber_loss,
                train_step_counter=self.train_step_counter)

        if len(self._mcfg.clip_layer_names)>0:
            self.agent._clip_layer_names = self._mcfg.clip_layer_names
            self.agent._clip_norm_value = self._mcfg.gradient_clipping
        else:
            self.agent._clip_layer_names = []
            self.agent._clip_norm_value = None

        self.agent.initialize()
        self.agent.train = common.function(self.agent.train)

    def _is_prioritized_replay(self) -> bool:
        return self._mcfg.replay_sampler.lower() == 'prioritized'

    def _per_first(self, value):
        value = tf.convert_to_tensor(value)
        value = tf.reshape(value, [tf.shape(value)[0], -1])
        return value[:, 0]

    def _per_beta(self, step:int) -> float:
        progress = min(1.0, step / max(1, self._mcfg.num_iterations))
        return self._mcfg.per_beta_start + progress * (self._mcfg.per_beta_end - self._mcfg.per_beta_start)

    def _per_weights(self, sample_info, beta:float):
        if not hasattr(sample_info, 'probability'):
            return None
        probs = tf.cast(self._per_first(sample_info.probability), tf.float32)
        table_size = tf.cast(self._per_first(sample_info.table_size), tf.float32)

        weights = tf.pow(tf.maximum(table_size * probs, 1e-12), -beta)
        return weights / tf.reduce_max(weights)

    def _update_per_priorities(self, sample_info, train_loss) -> None:
        if not hasattr(sample_info, 'key'):
            return
        keys = self._per_first(sample_info.key)

        td_error = tf.abs(tf.cast(train_loss.extra.td_error, tf.float32))
        td_error = tf.reshape(td_error, [tf.shape(td_error)[0], -1])
        priorities = tf.reduce_max(td_error, axis=1) + self._mcfg.per_priority_epsilon
        priorities = tf.clip_by_value(
            priorities,
            self._mcfg.per_priority_epsilon,
            self._mcfg.per_priority_max)

        self.replay_buffer.update_priorities(keys, tf.cast(priorities, tf.float64))

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
            best_return_var = tf.Variable(float('-inf'), dtype=tf.float32, name="best_return")
            self.best_return_ckpt_var = best_return_var
            self.ckpt = tf.train.Checkpoint(
                step=tf.Variable(1),
                agent=self.agent,
                policy=self.agent.policy,
                replay_buffer=self.replay_buffer,
                global_step=self.train_step_counter,
                custom_variable=avg_return_var,
                best_return=best_return_var
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

    def warm_start_weights(self, source_checkpoint_dir: str, ckpt_name: str = None, reset_counter:bool = False) -> None:
        """Restore Q-network and target network weights from a previous run.
        Resets train_step_counter and optimizer.iterations so epsilon and LR
        schedule both start from scratch — only the learned weights carry over."""
        warm_ckpt = tf.train.Checkpoint(agent=self.agent)
        manager = tf.train.CheckpointManager(warm_ckpt, source_checkpoint_dir, max_to_keep=None)

        if ckpt_name:
            source = "{}/{}".format(source_checkpoint_dir, ckpt_name)
            if source not in manager.checkpoints:
                print("Warm start: {} not found, falling back to latest".format(source))
                source = manager.latest_checkpoint
        else:
            source = manager.latest_checkpoint

        if source is None:
            print("Warm start: no checkpoint found in {}".format(source_checkpoint_dir))
            return

        print("Warm start: optimizer.iterations before restore: {}".format(self.optimizer.iterations.numpy()))

        warm_ckpt.restore(source).expect_partial()

        # optimizer.iterations is embedded in agent and gets restored with it;
        # always reset it so the LR schedule (cosine decay) starts from step 0.
        self.optimizer.iterations.assign(0)
        print("Warm start: optimizer.iterations after restore+reset: {}".format(self.optimizer.iterations.numpy()))

        # Print Adam slot (m/v) stats for the first layer — TF2-compatible.
        first_var = self.agent._q_network.trainable_variables[0]
        base_name = first_var.name.rsplit(':', 1)[0].split('/')[-1]
        slot_vars = [v for v in self.optimizer.variables()
                     if base_name in v.name and v.shape.rank > 0
                     and 'iteration' not in v.name.lower()]
        for v in slot_vars[:4]:
            print("  slot '{}': mean={:.6f}, max={:.6f}".format(
                v.name, float(tf.reduce_mean(v)), float(tf.reduce_max(v))))

        # After restoring checkpoint, snapshot weights
        self.prev_weights = {v.name: v.numpy().copy() for v in self.agent._q_network.trainable_variables}

        self.prev_weights_collection = {name: [] for name in self.prev_weights.keys()}
        self.prev_weights_collection['Step'] = []

        if reset_counter:
            self.train_step_counter.assign(0)

        print("Warm-started from: {} {}".format(source, "(step counter reset to 0)" if reset_counter else ""))

    def evaluate_chkpt(self, evt_ckpnt:str) -> list:
        """Restore and evaluate checkpoint"""
        if self.debug:
            print("Restore Ckpt from: {}".format(evt_ckpnt))

        # Restore only agent + metadata; replay_buffer is excluded because its capacity
        # in the saved checkpoint may differ from the eval-mode buffer (shape mismatch).
        eval_ckpt = tf.train.Checkpoint(
            agent=self.agent,
            global_step=self.train_step_counter,
            custom_variable=self.ckpt.custom_variable,
        )
        eval_ckpt.restore(evt_ckpnt).expect_partial()
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

    def get_current_lr(self) -> float:
        """Return the current learning rate regardless of whether it is a
        fixed float or a callable schedule (e.g. CosineDecay)."""
        lr = self.optimizer.learning_rate
        if callable(lr):
            return float(lr(self.optimizer.iterations))
        return float(lr)

    def train(self) -> None:

        #Set CTRL+C handler
        signal.signal(signal.SIGINT, ModelTrain.handler)

        if self._mcfg.if_evaluate_chkpoint:
            self.evaluate()
            return

        print("Start training.....")



        if self.debug:
            print_summary(self.q_net)

        train_collect_policy = py_tf_eager_policy.PyTFEagerPolicy(self.agent.collect_policy, use_tf_function=True)
        train_driver = py_driver.PyDriver(
            env=self._train_env, #self._train_py_env
            policy=train_collect_policy,
            observers=[self.rb_observer],
            end_episode_on_boundary=True,
            max_steps=self._mcfg.train_driver_max_step,
            max_episodes=0)

        #policy_state = train_collect_policy.get_initial_state(self._train_py_env.batch_size)
        policy_state = train_collect_policy.get_initial_state(self._train_env.batch_size)
        print(self._train_env.batch_size)

        #put initial number of records to buffer
        train_time_step = self.collect_episode(self._train_env, #self._train_py_env,
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
        lrn_rates = []
        qvalue_diag = []
        replay_diag = []

        loss_counter = 0.0

        # Restore best_return from the checkpoint so a resumed session does not
        # trivially overwrite the true best checkpoint with its first eval.
        if self.best_return_ckpt_var is not None:
            self.best_return = float(self.best_return_ckpt_var.numpy())
        else:
            self.best_return = float('-inf')
        self.no_improve_evals = 0
        print("Initial best_return: {:.2f}".format(self.best_return))

        mutils.param_gradients(0, self.q_net, grads)
        mutils.log_weight_histograms(0, self.q_net, self.tb_writer)

        if self._mcfg.dynamic_lrn_rate:
            #print("LRate {} -> {:.5f}".format(0, self.get_current_lr()))
            lrn_rates.append([0, self.get_current_lr()])

        iterator = iter(self.replay_buffer.as_dataset(sample_batch_size=self._mcfg.batch_size, num_steps=self._mcfg.sequence_length))

        for _ in range(self._mcfg.num_iterations):
            # Collect a few episodes using collect_policy and save to the replay buffer.
            #changed num_steps = batch_size to 0 Use to episodes = 1 instead 0
            #modified - no agent - random

            train_time_step, policy_state = train_driver.run(
                time_step=train_time_step,
                policy_state=policy_state,
            )

            num_frames = self.replay_buffer.num_frames()

            # Use data from the buffer and update the agent's network.
            trajectories, sample_info = next(iterator)
            weights = None
            beta = float('nan')
            if self._is_prioritized_replay():
                step_before = int(self.agent.train_step_counter.numpy())
                beta = self._per_beta(step_before)
                weights = self._per_weights(sample_info, beta)
                train_loss = self.agent.train(experience=trajectories, weights=weights)
                self._update_per_priorities(sample_info, train_loss)
            else:
                train_loss = self.agent.train(experience=trajectories)
            loss_counter += train_loss.loss

            reward_per_batch = (np.sum(trajectories.reward.numpy())/self._mcfg.batch_size)

            step = self.agent.train_step_counter.numpy()

            # Decay epsilon each step
            epsilon = self._mcfg.epsilon_end + (self._mcfg.epsilon_start - self._mcfg.epsilon_end) * math.exp(-self._mcfg.epsilon_decay * step)
            self.agent.collect_policy._epsilon = epsilon  # inject updated value

            if step > 0 and step % self._mcfg.log_interval == 0 and self.debug:
                if self._mcfg.log_interval <= self._mcfg.log_loss_interval:
                    print('step = {0}: loss = {1:0.5f} Reward: {2:0.3f} ε={3:.4f} Sec. {4} Frames: {5}'.format(step,
                            loss_counter/self._mcfg.log_interval, reward_per_batch, epsilon, (datetime.now()-tm_start).seconds, num_frames))
                else:
                    print('step = {0}: loss = {1:0.3f} Reward: {2:0.3f} ε={3:.4f} Sec. {4} Frames: {5}'.format(step,
                            train_loss.loss, reward_per_batch, epsilon, (datetime.now()-tm_start).seconds, num_frames))

            if step > 0 and step % self._mcfg.log_loss_interval == 0:
                avg_loss = loss_counter/self._mcfg.log_loss_interval
                loss_list.append([step, avg_loss])
                loss_counter = 0.0

                mutils.log_scalar(step, "Loss/train", avg_loss, self.tb_writer)
                mutils.param_gradients(step, self.q_net, grads, agent=self.agent)
                mutils.log_weight_histograms(step, self.q_net, self.tb_writer)
                qvalue_diag.append(self.collect_qvalue_diagnostics(int(step), trajectories))
                replay_diag.append(self.collect_replay_diagnostics(int(step), sample_info, num_frames, weights=weights, beta=beta))

                if self._mcfg.dynamic_lrn_rate:
                    #print("LRate {} -> {:.5f}".format(step, self.get_current_lr()))
                    lrn_rates.append([step, self.get_current_lr()])

            if step > 0 and step % self._mcfg.eval_interval == 0:
                avg_return = self.compute_avg_return(self._tf_eval_env, self.agent.policy, self._mcfg.num_eval_episodes)
                returns.append(avg_return)
                mutils.log_scalar(step, "Return/eval", avg_return, self.tb_writer)
                if self.debug:
                    print('---> Step = {0}: Average Return = {1:0.2f} All: {2}'.format(step, avg_return, returns))

                if self.prev_weights:
                    # After N training steps with warm up start:
                    self.prev_weights_collection['Step'].append(step)
                    for v in self.agent._q_network.trainable_variables:
                        delta = np.linalg.norm(v.numpy() - self.prev_weights[v.name])
                        self.prev_weights_collection[v.name].append(delta)

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
                    self.best_return_ckpt_var.assign(avg_return)
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
            print('---> Step = {0}: Average Return = {1:0.2f} All: {2}'.format(step, avg_return, returns))

        # final eval may itself be a new best
        if self.ckpt and avg_return >= self.best_return + self._mcfg.early_stop_min_delta:
            self.best_return = avg_return
            self.ckpt.custom_variable.assign(avg_return)
            self.best_return_ckpt_var.assign(avg_return)
            self.best_ckpt_manager.save()

        if self.best_ckpt_manager and self.best_ckpt_manager.latest_checkpoint:
            print("Best checkpoint: {} (avg return {:0.2f})".format(
                self.best_ckpt_manager.latest_checkpoint, self.best_return))

        if self.prev_weights:
            # After N training steps with warm up start:
            self.prev_weights_collection['Step'].append(step)
            for v in self.agent._q_network.trainable_variables:
                delta = np.linalg.norm(v.numpy() - self.prev_weights[v.name])
                self.prev_weights_collection[v.name].append(delta)
                print(f"{v.name}: delta_norm={delta:.4f}")

            mutils.save_weights(self.prev_weights_collection, self._mcfg.weights_file)

        if self.ckpt:
            self.ckpt.step.assign_add(1)
            self.ckpt.custom_variable.assign(avg_return)
            sv_folder = self.ckpt_manager.save()
            if self.debug:
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), sv_folder))

        mutils.save_results(self._mcfg.results_file, returns)
        mutils.save_info2cvs(self._mcfg.loss_file, loss_list, ["Step", "Loss"])

        if self._mcfg.dynamic_lrn_rate:
            mutils.save_info2cvs(self._mcfg.lrnrt_file, lrn_rates, ["Step", "LrnRate"], sformat="{:.6f}")

        prm_headrs = mutils.param_names(self.q_net)
        mutils.save_info2cvs(self._mcfg.gradient_file, grads, prm_headrs)
        mutils.save_info2cvs(self._mcfg.qvalue_file, qvalue_diag, self.qvalue_diag_headers(), sformat="{:.6f}")
        mutils.save_info2cvs(self._mcfg.replay_diag_file, replay_diag, self.replay_diag_headers(), sformat="{:.6f}")

        mutils.save_info2list(self._mcfg.all_results_file, returns, name=self._mcfg.data_idx)

        #['Date', 'Name', 'Duration','NumIterations', 'BatchSize','UpTau', 'UpPrd', 'LrnRate', 'Gamma', 'Eps_Start', 'Eps_End', 'Eps_decay', 'GradClip', 'InitRecords', 'KernelInitType']

        mutils.save_parameters(tm_start, self._mcfg.data_idx, [self._mcfg.num_iterations, self._mcfg.batch_size, self._mcfg.target_update_tau,
                        self._mcfg.target_update_period, self._mcfg.lrn_rate, self._mcfg.gamma,
                        self._mcfg.epsilon_start, self._mcfg.epsilon_end, self._mcfg.epsilon_decay,
                        self._mcfg.gradient_clipping, self._mcfg.num_initial_records, self._mcfg.kernel_init_type],
                        self._mcfg.layer_sz,
                        self._mcfg.clip_layer_names)

        print("Training finished..... {}".format(datetime.now() - tm_start))
        print(returns)


def _suppress_pool_del_oserror(unraisable):
    import errno
    if (unraisable.exc_type is OSError and
            unraisable.exc_value.errno == errno.EBADF and
            'Pool' in type(unraisable.object).__qualname__):
        return
    sys.__unraisablehook__(unraisable)

if __name__ == '__main__':
    sys.unraisablehook = _suppress_pool_del_oserror

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]  # MB
        )

    cfg = ModelCfg()

    attempt = 13
    label = None
    step_idx = None
    warm_start_label = None
    warm_start_ckpt = None

    for cmd in sys.argv:
        if cmd.find("--step=") >= 0:
            step_idx = cmd.split('=')[1]
            if len(step_idx)==0:
                print("No folder for evaluation")
                exit()

        if cmd.find("--label=") >= 0:
            label = cmd.split('=')[1]

        if cmd.find("--warm_start=") >= 0:
            warm_start_label = cmd.split('=')[1]

        if cmd.find("--warm_start_ckpt=") >= 0:
            warm_start_ckpt = "ckpt-" + cmd.split('=')[1]

    if step_idx:
        if not label or len(step_idx)==0:
            print("No folder for evaluation")
            exit()

        print("Evaluate parameters: {} {}".format(label, step_idx))

        cfg.data_idx = label
        cfg.evaluate_chkpoint = step_idx

        mdl = ModelTrain(cfg=cfg)
        mdl.debug = True
        mdl.initialise()
        mdl.evaluate()
        exit()


    #for kernel_init_type in ['VarianceScaling', 'GlorotNormal', 'GlorotUniform']:
    for lrn_rate in [0.000025, 0.000015, 0.00005]:
        for cosine_decay_alpha in [0.1, 0.02]:
            lbl = label if label else "LL_{}".format(1 + attempt)
            cfg.data_idx = lbl
            # LL_8/9/10: keep cosine_decay_steps=200K (helped LL_5/6 find good policy
            # earlier), revert tau to 0.001 and period to 15 — tau=0.005 in LL_5/6
            # accelerated Q-value divergence (QStd hit 132/191) vs 0.002 in LL_2-4.
            # More stable target network to slow divergence. LR sweep: 2.5e-5, 1.5e-5, 5e-5.
            # Alpha sweep: 0.1 vs 0.02 LR floor (fraction of peak held after decay).
            cfg.replay_sampler = 'uniform'
            cfg._lrn_rate = lrn_rate
            cfg._dynamic_lrn_rate = True
            cfg._cosine_decay_steps = 200000  # decay to floor by 200K; hold floor for remaining 50K
            cfg._cosine_decay_alpha = cosine_decay_alpha
            cfg._num_initial_records = 25000
            cfg.num_iterations = 250000

            cfg._epsilon_start = 1.0
            cfg._epsilon_decay = 0.000008
            cfg._epsilon_end = 0.01

            cfg._num_eval_episodes = 30
            cfg._early_stop_enabled = True
            cfg._early_stop_patience = 6   # ~120K steps; halved for 250K run
            cfg._early_stop_min_delta = 5.0
            cfg._early_stop_target = 200.0

            cfg._target_update_tau = 0.001   # reverted: 0.005 accelerated Q-value divergence
            cfg._target_update_period = 15

            cfg._clip_layer_names = []
            cfg._gradient_clipping = 1.5
            cfg.kernel_init_type = 'GlorotNormal'

            if warm_start_label:
                # LL_49: same warm-start fine-tune recipe as LL_46/47/48, with the
                # gradient clip loosened to break the plateau. LL_46/47/48 all
                # converged to the same loss (~0.84) and the same negative-mean,
                # ~100-ceiling returns (never solved); the 0.3 per-layer clip bound
                # on 100% of steps, so updates were fixed-magnitude regardless of the
                # true gradient. Loosen the clip to let real gradient magnitude
                # through. lr and num_iterations match LL_48 so the clip is the only
                # changed variable. Watch for LL_42/43-style divergence (loss down,
                # return collapse, monotonic weight drift).
                cfg._epsilon_start = 0.1      # refill buffer with some diversity
                cfg._epsilon_end = 0.05       # keep a floor — avoid greedy collapse
                cfg._epsilon_decay = 0.00002  # ~reaches floor over the run
                cfg._lrn_rate = 0.00001       # 1e-5 — match LL_48 (isolate clip change)
                cfg.num_iterations = 600000
                cfg._num_initial_records = 5000
                cfg._gradient_clipping = 2.0  # was 0.3 (LL_46-48); old clip bound 100% of steps
                cfg._dynamic_lrn_rate = False

            mdl = ModelTrain(cfg=cfg)
            mdl.debug = True
            mdl.initialise()

        if warm_start_label:
            # LL_14: warm-start from LL_11_best. Clip tightened to 1.0 (was 2.0 in
            # LL_11/13) to control Q-value divergence — QMin was spiking to -500+
            # across every warm-start run with clip=2.0. Everything else unchanged
            # so clip is the single variable under test.
            cfg._epsilon_start = 0.1      # refill buffer with some diversity
            cfg._epsilon_end = 0.05       # keep a floor — avoid greedy collapse
            cfg._epsilon_decay = 0.00002  # ~reaches floor over the run
            cfg._lrn_rate = 0.00001       # 1e-5 fresh start
            cfg.num_iterations = 500000
            cfg._num_initial_records = 5000
            cfg._gradient_clipping = 2.0
            cfg._dynamic_lrn_rate = False
            cfg._early_stop_enabled = False
            cfg._early_stop_min_delta = 1.0

        mdl = ModelTrain(cfg=cfg)
        mdl.debug = False
        mdl.initialise()

        if warm_start_label:
            src_dir = cfg.data_folder + 'multi_checkpoint_{}'.format(warm_start_label)
            mdl.warm_start_weights(src_dir, warm_start_ckpt)

        mdl.train()
        attempt += 1

        if warm_start_label:
            break  # warm-start runs once; the LR sweep is irrelevant when params are overridden
