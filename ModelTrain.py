#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import signal
from datetime import datetime

import math

import numpy as np
import reverb

import gymnasium as gym

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout

from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import tensor_spec
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import sequential
from tf_agents.utils import common
from tf_agents.policies import py_tf_eager_policy, random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.drivers import py_driver
from tf_agents.networks.layer_utils import print_summary

from ModelCfg import ModelCfg
import ModelUtils as mutils
from gym_wrap import GymnasiumWrapper

class SelectiveClipDqnAgent(dqn_agent.DqnAgent):
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
            loss_info = self._loss(experience, weights=weights, training=True)

        variables = self._q_network.trainable_variables
        gradients = tape.gradient(loss_info.loss, variables)

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
        """"""
        self._mcfg = cfg

        #self._train_py_env = suite_gym.load(self._mcfg.env_name)
        #self._train_py_env.reset()

        #self._eval_py_env = suite_gym.load(self._mcfg.env_name)
        #self._eval_py_env.reset()

        #self._train_env = tf_py_environment.TFPyEnvironment(self._train_py_env)
        #self._eval_env = tf_py_environment.TFPyEnvironment(self._eval_py_env)

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
        self.ckpt_restored=False

        self._debug = False

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

            while not time_step.is_last():
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

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

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
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self._mcfg.lrn_rate*5,   # start higher  0.0001 than your current 0.00002
                decay_steps=self._mcfg.num_iterations,
                alpha=0.05,                        # floor = 5% of initial = 5e-6, not 1e-5
                warmup_target=self._mcfg.lrn_rate,
                warmup_steps=self._mcfg.num_iterations * 0.1
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

    def init_train_data(self) -> None:
        """Prepre replay buffer"""
        replay_buffer_signature = tensor_spec.from_spec(self.agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

        table_name = 'uniform_table'
        table = reverb.Table(
            table_name,
            max_size=self._mcfg.replay_buffer_capacity,
            sampler=reverb.selectors.Prioritized(priority_exponent=0.6), #reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=replay_buffer_signature
            )

        reverb_server = reverb.Server([table])

        self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            table_name=table_name,
            sequence_length=self._mcfg.sequence_length,
            dataset_buffer_size=self._mcfg.batch_size*self._mcfg.sequence_length,
            local_server=reverb_server)

        self.rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            self.replay_buffer.py_client,
            table_name,
            sequence_length=self._mcfg.sequence_length)

    def init_checkpoints(self) -> None:
        if self._mcfg.checkpoint_dir:
            self.ckpt = tf.train.Checkpoint(
                step=tf.Variable(1),
                agent=self.agent,
                policy=self.agent.policy,
                replay_buffer=self.replay_buffer,
                global_step=self.train_step_counter
            )
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self._mcfg.checkpoint_dir, max_to_keep=self._mcfg.ckpt_max_to_keep)
            ckpt_mng_last = self.ckpt_manager.latest_checkpoint

            if self.debug:
                print("Available checkpoints: {}".format(self.ckpt_manager.checkpoints))

            if ckpt_mng_last is not None:
                print("Restore Ckpt from: {}".format(ckpt_mng_last))
                self.ckpt.restore(ckpt_mng_last).expect_partial()
                self.ckpt_restored=True
            else:
                print("No checkpoints")

            if self.debug:
                print("Loaded checkpoint from: {} Step: {} Save counter: {}".format(self._mcfg.checkpoint_dir, self.train_step_counter.numpy(), self.ckpt.save_counter.numpy()))

    def evaluate(self) -> None:
        print("Start evaluation.....")

        tm_start = datetime.now()
        self.debug = True

        if self._mcfg.if_evaluate_chkpoint:
            evt_ckpnt = "{}/{}".format(self._mcfg.checkpoint_dir, self._mcfg.evaluate_chkpoint)
            print(evt_ckpnt)
            ckpt_mng_last = evt_ckpnt if evt_ckpnt in self.ckpt_manager.checkpoints else self.ckpt_manager.latest_checkpoint

            print("Available checkpoints: {}".format(self.ckpt_manager.checkpoints))
            if ckpt_mng_last is None:
                print("No available checkpoints. Stop eveluation.")
                return

            print("Restore Ckpt from: {}".format(ckpt_mng_last))
            self.ckpt.restore(ckpt_mng_last).expect_partial()

            eval_result = []
            for _ in range(3):
                eval_result.append(self.compute_avg_return(self._tf_eval_env, self.agent.policy, self._mcfg.num_eval_episodes))
                print(eval_result)

        print("Evaluation finished..... {}".format(datetime.now() - tm_start))

    def get_current_lr(self) -> float:
        """Return the current learning rate regardless of whether it is a
        fixed float or a callable schedule (e.g. CosineDecay)."""
        lr = self.optimizer.learning_rate
        if callable(lr):
            return float(lr(self.optimizer.iterations))
        return float(lr)

    def train(self) -> None:
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

        tm_start = datetime.now()

        loss_list = []
        grads = []
        lrn_rates = []

        loss_counter = 0.0

        mutils.param_gradients(0, self.q_net, grads)
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
            trajectories, _ = next(iterator)
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
                loss_list.append([step, loss_counter/self._mcfg.log_loss_interval])
                loss_counter = 0.0

                mutils.param_gradients(step, self.q_net, grads, agent=self.agent)

                if self._mcfg.dynamic_lrn_rate:
                    #print("LRate {} -> {:.5f}".format(step, self.get_current_lr()))
                    lrn_rates.append([step, self.get_current_lr()])

            if step > 0 and step % self._mcfg.eval_interval == 0:
                avg_return = self.compute_avg_return(self._tf_eval_env, self.agent.policy, self._mcfg.num_eval_episodes)
                returns.append(avg_return)
                if self.debug:
                    print('---> Step = {0}: Average Return = {1:0.2f} All: {2}'.format(step, avg_return, returns))

            if step > 0 and step % self._mcfg.episode_for_checkpoint == 0 and self.ckpt:
                self.ckpt.step.assign_add(1)
                sv_folder = self.ckpt_manager.save()
                if self.debug:
                    print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), sv_folder))

            if self.finish_train:
                break

        avg_return = self.compute_avg_return(self._tf_eval_env, self.agent.policy, self._mcfg.num_eval_episodes)
        returns.append(avg_return)
        if self.debug:
            print('---> Step = {0}: Average Return = {1:0.2f} All: {2}'.format(step, avg_return, returns))

        if self.ckpt:
            self.ckpt.step.assign_add(1)
            sv_folder = self.ckpt_manager.save()
            if self.debug:
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), sv_folder))

        mutils.save_results(self._mcfg.results_file, returns)
        mutils.save_info2cvs(self._mcfg.loss_file, loss_list, ["Step", "Loss"])

        if self._mcfg.dynamic_lrn_rate:
            mutils.save_info2cvs(self._mcfg.lrnrt_file, lrn_rates, ["Step", "LrnRate"], sformat="{:.5f}")

        prm_headrs = mutils.param_names(self.q_net)
        mutils.save_info2cvs(self._mcfg.gradient_file, grads, prm_headrs)

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


if __name__ == '__main__':
    cfg = ModelCfg()

    attempt = 1
    """
    for eps_decay in [0.00001, 0.00002, 0.00003]:
        for grad_clip_names in [["LYR_"], ["LYR_", "Output"]]:
            for grad_val in [0.4, 0.5, 0.6]:
                lbl = "LL_{}".format(attempt)
                cfg.data_idx = lbl
                cfg._epsilon_decay = eps_decay
                cfg._clip_layer_names = grad_clip_names
                cfg._gradient_clipping = grad_val

                mdl = ModelTrain(cfg=cfg)
                mdl.initialise()
                mdl.train()
                attempt += 1
    """

    """
    for lrn_rate in [0.00001, 0.00002, 0.00003, 0.0001]:
        for grad_clip_names in [["LYR_"], ["LYR_", "Output"]]:
            for target_update_tau in [0.01, 0.02, 0.03]:
                for target_update_period in [10, 15, 20]:
                    lbl = "LL_{}".format(attempt+40)
                    cfg.data_idx = lbl
                    cfg._epsilon_decay = 0.00002
                    cfg._clip_layer_names = grad_clip_names
                    cfg._gradient_clipping = 0.5
                    cfg._target_update_tau = target_update_tau
                    cfg._target_update_period = target_update_period

                    mdl = ModelTrain(cfg=cfg)
                    mdl.initialise()
                    mdl.train()
                    attempt += 1
    """
    """
    for kernel_init_type in ['VarianceScaling', 'GlorotNormal', 'GlorotUniform']:
        for lrn_rate in [0.00001, 0.00002, 0.00003, 0.0001]:
            lbl = "LL_{}".format(attempt+120)
            cfg.data_idx = lbl
            cfg._epsilon_decay = 0.00002
            cfg._clip_layer_names = ["LYR_"]
            cfg._gradient_clipping = 0.5
            cfg._target_update_tau = 0.01
            cfg._target_update_period = 10
            cfg.kernel_init_type = kernel_init_type
            cfg._lrn_rate = lrn_rate

            mdl = ModelTrain(cfg=cfg)
            mdl.initialise()
            mdl.train()
            attempt += 1
    """

    for cmd in sys.argv:
        if cmd.find("--evaluate=") >= 0:
            data_idx = cmd.split('=')[1]
            if len(data_idx)==0:
                print("No folder for evaluation")
                exit()

            cfg.data_idx = data_idx
            cfg.evaluate_chkpoint = 'last'
            mdl = ModelTrain(cfg=cfg)
            mdl.initialise()
            mdl.evaluate()
            exit()


    #for kernel_init_type in ['VarianceScaling', 'GlorotNormal', 'GlorotUniform']:
    #for grad_clip_names in [["LYR_"]]:
    #for target_update_tau in [0.005]:
    lbl = "LL_{}".format(attempt+6)
    cfg.data_idx = lbl
    cfg._lrn_rate        = 0.00005   # halved — reduce clipped gradient pressure
    #cfg._lrn_rate         = 0.0001        # actual cosine start (was likely 0.00002)
    cfg._dynamic_lrn_rate = True          # keep cosine, but fix alpha:
    # In init_agent: alpha=0.05 instead of 0.1 (floor at 5000e-4, not 1e-5)

    cfg._epsilon_start = 1.0
    cfg._epsilon_decay   = 0.00003   # slower decay for longer run
    #cfg._epsilon_decay    = 0.00006       # reach ~ε_end by step ~80k
    cfg._epsilon_end      = 0.01

    cfg._target_update_tau = 0.002   # softer than LL_2's 0.01
    cfg._target_update_period = 15   # back to default
    #cfg._target_update_tau    = 0.01      # faster target tracking
    #cfg._target_update_period = 10

    #cfg._num_initial_records  = 20000     # more warm-up before learning start

    cfg._clip_layer_names = ["LYR_"]  # set [] for disabling gradient clipping by layer
    #cfg._gradient_clipping = 1.0     # unblock the hidden layers
    cfg._gradient_clipping = 0.5
    cfg.kernel_init_type = 'GlorotNormal'

    #cfg.num_iterations = 1000

    mdl = ModelTrain(cfg=cfg)
    mdl.debug = True
    mdl.initialise()
    mdl.train()
    attempt += 1
