#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Important: Tensorflow version 2.15

import os
import sys

import tensorflow as tf

class ModelCfg(object):
    """Configuration parameters for model"""

    def __init__(self) -> None:
        self._env_name = 'LunarLander-v3' #'LunarLander-v2'
        #max number of iterations
        self._num_iterations = 1200000 if self.env_name == 'LunarLander-v3' else 25000
        #number collected episodes per iteration
        self._collect_episode_per_iteration = 2
        #replay buffer capacity
        self._replay_buffer_capacity = self.num_iterations*2 #if self.num_iterations <= 120000 else self.num_iterations + 50000
        #number initially generated records in reply buffer
        self._num_initial_records = 25000

        #training batch size
        self._batch_size = 256
        #generate by one tragectory each request
        self._train_driver_max_step=1

        #number evaluetion epiodes
        self._num_eval_episodes = 10
        self._sequence_length = 2
        self._n_step_update = self.sequence_length - 1


        #Intervals
        self._eval_interval = 20000 # Evaluation intervall
        self._log_interval = 5000 # Print output information each n step
        self._log_loss_interval = self.log_interval

        #checkpoints
        self._ckpt_max_to_keep = 20   #max number of checkpoints
        self._episode_for_checkpoint = self.eval_interval  #recored checkpoint each b steps

        #Factor for soft update of the target networks.
        self._target_update_tau=0.001
        #Period for soft update of the target networks.
        self._target_update_period=15

        #
        #Important parameters
        #
        #lerning rate - Adam optimizer parameter (0.001 - 0.0001)
        #Note: on my understanding there is size of step used for gradient calculation
        self._lrn_rate=0.00002
        self._dynamic_lrn_rate = False

        #The discount factor (γ) of future rewards - gamma (0.9-1.0)
        self._gamma=0.99

        #Epsilon-Greedy Exploration: A strategy used during training to balance exploration
        #(taking random actions to discover new possibilities) and exploitation (taking the action with the highest predicted Q-value).
        #The probability of taking a random action, epsilon, typically decays over time. 
        self._epsilon_start  = 1.0
        self._epsilon_end    = 0.02
        # controls how fast it falls
        self._epsilon_decay  = 0.00002

        #Geometrically, this constrains the gradient to lie within a hypersphere of radius τ
        #centered at the origin. In 2D, this is a circle; in 3D, a sphere.
        # Any gradient landing outside this sphere gets projected back onto its surface by scaling, not by truncating individual components.
        self._gradient_clipping = 0.5

        # Layers you want clipped — match by name prefix
        # hidden layers only; excludes "Input" and "Output"
        # Set to [] if you do not want to apply layes clipping at all
        self._clip_layer_names = ["LYR_"]

        #Output layes bias initialization
        self._bias_lyr_out = tf.keras.initializers.Constant(0)
        self._kernel_init_lyr_out = tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03)

        #['VarianceScaling', 'GlorotNormal', 'GlorotUniform']
        self._kernel_init_type = 'VarianceScaling'

        #layers size
        self.layer_sz = [256, 256] #[128, 128]

        self._data_folder = './data/'
        self._data_idx = "ll_01"

        self._evaluate_chkpoint = ''


    def _init_out_files(self) -> None:
        self._checkpoint_dir = self.data_folder + 'multi_checkpoint_{}'.format(self.data_idx)
        self._results_file = self.data_folder+'results_{}.csv'.format(self.data_idx)

        self._all_results_file = self.data_folder+'results.csv'

        self._loss_file = self.data_folder+"loss_{}.csv".format(self.data_idx)
        self._lrnrt_file = self.data_folder+"lrnrates_{}.csv".format(self.data_idx)
        self._gradient_file = self.data_folder+"pgradients_{}.csv".format(self.data_idx)


    def _init_layer_depends(self) -> None:
        #bias initialization
        self._bias = [tf.keras.initializers.Constant(0.0)] * len(self.layer_sz)

        #dropout layers initialization
        self._dropout = [0.0] * len(self.layer_sz)

        """
        self._kernel_init = [
                    tf.keras.initializers.VarianceScaling(
                        scale=1.0,
                        mode='fan_in',
                        distribution='truncated_normal')
                ] * len(self.layer_sz)
        """

        #Change layer intialization
        #Also need try
        #GlorotUniform
        #GlorotNormal
        self._kernel_init = []
        for k in range(len(self.layer_sz)):
            if self.kernel_init_type == 'VarianceScaling':
                self._kernel_init.append(
                    tf.keras.initializers.VarianceScaling(
                        scale=1.0,
                        mode='fan_in',
                        distribution='truncated_normal')
                )
            elif self.kernel_init_type == 'GlorotUniform':
                self._kernel_init.append(
                    tf.keras.initializers.GlorotUniform()
                )
            elif self.kernel_init_type == 'GlorotNormal':
                self._kernel_init.append(
                    tf.keras.initializers.GlorotNormal()
                )

    @property
    def dynamic_lrn_rate(self) -> bool:
        return self._dynamic_lrn_rate

    @property
    def checkpoint_dir(self) -> str:
        return self._checkpoint_dir

    @property
    def all_results_file(self) -> str:
        return self._all_results_file

    @property
    def results_file(self) -> str:
        return self._results_file

    @property
    def loss_file(self) -> str:
        return self._loss_file

    @property
    def lrnrt_file(self) -> str:
        return self._lrnrt_file

    @property
    def gradient_file(self) -> str:
        return self._gradient_file

    @property
    def data_folder(self) -> str:
        return self._data_folder

    @data_folder.setter
    def data_folder(self, didx:str) -> str:
        self._data_folder = didx
        self._init_out_files()

    @property
    def data_idx(self) -> str:
        return self._data_idx

    @data_idx.setter
    def data_idx(self, didx:str) -> str:
        self._data_idx = didx
        self._init_out_files()

    @property
    def bias(self) -> list:
        return self._bias

    @property
    def dropout(self) -> list:
        return self._dropout

    @property
    def bias_lyr_out(self) -> any:
        return self._bias_lyr_out

    @property
    def kernel_init_lyr_out(self) -> any:
        return self._kernel_init_lyr_out

    @property
    def kernel_init(self) -> list:
        return self._kernel_init

    @property
    def kernel_init_type(self) -> str:
        return self._kernel_init_type

    @kernel_init_type.setter
    def kernel_init_type(self, val:str) -> None:
        self._kernel_init_type = val if val in ['VarianceScaling', 'GlorotNormal', 'GlorotUniform'] else 'VarianceScaling'
        self._init_layer_depends()

    @property
    def clip_layer_names(self) -> list:
        return self._clip_layer_names

    @property
    def layer_sz(self) -> list:
        return self._layer_sz

    @layer_sz.setter
    def layer_sz(self, val:list) -> None:
        self._layer_sz = val
        self._init_layer_depends()

    @property
    def lrn_rate(self) -> float:
        return self._lrn_rate

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def epsilon_start(self) -> float:
        return self._epsilon_start

    @property
    def epsilon_end(self) -> float:
        return self._epsilon_end

    @property
    def epsilon_decay(self) -> float:
        return self._epsilon_decay

    @property
    def gradient_clipping(self) -> float:
        return self._gradient_clipping

    @property
    def env_name(self) -> str:
        return self._env_name

    @property
    def num_iterations(self) -> int:
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, niter:int) -> None:
        self._num_iterations = niter
        #update reply buffer size too
        self._replay_buffer_capacity = self.num_iterations*2

    @property
    def collect_episode_per_iteration(self) -> int:
        return self._collect_episode_per_iteration

    @property
    def replay_buffer_capacity(self) -> int:
        return self._replay_buffer_capacity

    @property
    def num_initial_records(self) -> int:
        return self._num_initial_records

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def train_driver_max_step(self) -> int:
        return self._train_driver_max_step

    @property
    def num_eval_episodes(self) -> int:
        return self._num_eval_episodes

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def n_step_update(self) -> int:
        return self._n_step_update
    @property
    def eval_interval(self) -> int:
        return self._eval_interval

    @property
    def log_interval(self) -> int:
        return self._log_interval

    @property
    def log_loss_interval(self) -> int:
        return self._log_loss_interval

    @property
    def ckpt_max_to_keep(self) -> int:
        return self._ckpt_max_to_keep

    @property
    def episode_for_checkpoint(self) -> int:
        return self._episode_for_checkpoint

    @property
    def target_update_tau(self) -> float:
        return self._target_update_tau

    @property
    def target_update_period(self) -> int:
        return self._target_update_period

    @property
    def evaluate_chkpoint(self) -> str:
        return self._evaluate_chkpoint

    @evaluate_chkpoint.setter
    def evaluate_chkpoint(self, val:str) -> None:
        self._evaluate_chkpoint = val

    @property
    def if_evaluate_chkpoint(self) -> bool:
        return len(self._evaluate_chkpoint)>0
