#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import gymnasium as gym

from tf_agents.environments import utils
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec

class GymnasiumWrapper(py_environment.PyEnvironment):
    def __init__(self, gym_env):
        super().__init__()
        self._env = gym_env
        self._action_spec = self._convert_space_to_spec(self._env.action_space, 'action')
        self._observation_spec = self._convert_space_to_spec(self._env.observation_space, 'observation')
        self._state = None
        self._episode_ended = False

    def action_spec(self): return self._action_spec
    def observation_spec(self): return self._observation_spec

    def _convert_space_to_spec(self, space, name):
        if isinstance(space, gym.spaces.Box):
            return array_spec.ArraySpec(shape=space.shape, dtype=space.dtype, name=name)
        elif isinstance(space, gym.spaces.Discrete):
            return array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=space.n - 1, name=name)
        # Add support for other spaces (e.g., Tuple, Dict) here
        raise NotImplementedError(f"Unsupported space: {type(space)}")

    def _reset(self):
        obs, info = self._env.reset()
        self._state = obs
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._state = obs
        if terminated or truncated:
            self._episode_ended = True
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)

# Usage:
#train_py_env = suite_gym.load("CartPole-v1")
#tf_env = GymnasiumWrapper(train_py_env)

#raw_env = gym.make("LunarLander-v3") #("CartPole-v1")
#tf_env = GymnasiumWrapper(raw_env)

#utils.validate_py_environment(tf_env, episodes=5)
