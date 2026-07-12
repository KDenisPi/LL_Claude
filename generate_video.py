#!/usr/bin/env python3
"""Generate a recorded video of a LunarLander-v3 DDQN agent.

Loads a saved checkpoint, runs the greedy policy, and writes an mp4 using
gymnasium's RecordVideo wrapper (requires ffmpeg in the container).

Usage:
    # regular checkpoint
    python generate_video.py --label LL_23 --ckpt 31

    # best checkpoint
    python generate_video.py --label LL_23_best --ckpt 31 --episodes 3

    # custom output folder and resolution
    python generate_video.py --label LL_23_best --ckpt 31 --video-folder ./videos --width 1920 --height 1080
"""

import os
import sys
import argparse

import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


class _ResizeRenderWrapper(gym.Wrapper):
    """Upscale rendered frames to a target resolution before RecordVideo captures them."""
    def __init__(self, env, width, height):
        super().__init__(env)
        self._out_w = width
        self._out_h = height

    def render(self):
        frame = self.env.render()
        if frame is None or not isinstance(frame, np.ndarray):
            return frame
        return np.array(
            Image.fromarray(frame).resize((self._out_w, self._out_h), Image.LANCZOS)
        )

import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.utils import common

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ModelCfg import ModelCfg
from ModelTrain import SelectiveClipDqnAgent
from gym_wrap import GymnasiumWrapper


def _build_agent(tf_env, cfg):
    """Construct Q-network and agent matching the training architecture."""
    action_spec = tf_env.action_spec()
    num_actions = int(action_spec.maximum - action_spec.minimum + 1)

    layers = []
    for i, size in enumerate(cfg.layer_sz):
        layers.append(Dense(
            size,
            activation=tf.keras.activations.relu,
            name="LYR_{}".format(i),
            kernel_initializer=cfg.kernel_init[i],
            bias_initializer=cfg.bias[i],
        ))
    layers.append(Dense(
        num_actions,
        activation=None,
        name="Output",
        kernel_initializer=cfg.kernel_init_lyr_out,
        bias_initializer=cfg.bias_lyr_out,
    ))

    q_net = sequential.Sequential(
        layers,
        input_spec=tf_env.time_step_spec().observation,
        name="QNet",
    )

    train_step = tf.Variable(0)
    agent = SelectiveClipDqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lrn_rate),
        target_update_tau=cfg.target_update_tau,
        target_update_period=cfg.target_update_period,
        # gradient_clipping is handled selectively via clip_layer_names
        gradient_clipping=None if cfg.clip_layer_names else cfg.gradient_clipping,
        gamma=cfg.gamma,
        reward_scale_factor=cfg.reward_scale_factor,
        epsilon_greedy=0.0,  # greedy policy for video
        n_step_update=cfg.n_step_update,
        td_errors_loss_fn=common.element_wise_huber_loss,
        train_step_counter=train_step,
    )
    if cfg.clip_layer_names:
        agent._clip_layer_names = cfg.clip_layer_names
        agent._clip_norm_value = cfg.gradient_clipping
    agent.initialize()

    return agent, train_step


def _restore_checkpoint(agent, train_step, ckpt_path):
    """Restore agent weights only (no replay buffer)."""
    avg_return_var = tf.Variable(0.0)
    ckpt = tf.train.Checkpoint(
        agent=agent,
        global_step=train_step,
        custom_variable=avg_return_var,
    )
    ckpt.restore(ckpt_path).expect_partial()
    return float(avg_return_var.numpy()), int(train_step.numpy())


def _run_episode(tf_env, policy, max_steps=1000):
    time_step = tf_env.reset()
    total_return = 0.0
    steps = 0
    while not time_step.is_last() and steps < max_steps:
        action_step = policy.action(time_step)
        time_step = tf_env.step(action_step.action)
        total_return += float(time_step.reward.numpy()[0])
        steps += 1
    return total_return, steps


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--label", required=True,
                   help="Checkpoint directory label, e.g. LL_23 or LL_23_best")
    p.add_argument("--ckpt", required=True,
                   help="Checkpoint number, e.g. 31")
    p.add_argument("--video-folder", default="./videos",
                   help="Output directory for the video (default: ./videos)")
    p.add_argument("--episodes", type=int, default=1,
                   help="Number of episodes to record (default: 1)")
    p.add_argument("--data", default="./data",
                   help="Data root directory (default: ./data)")
    p.add_argument("--max-steps", type=int, default=1000,
                   help="Max steps per episode safety limit (default: 1000)")
    p.add_argument("--width", type=int, default=1280,
                   help="Video frame width in pixels (default: 1280)")
    p.add_argument("--height", type=int, default=720,
                   help="Video frame height in pixels (default: 720)")
    args = p.parse_args()

    ckpt_path = os.path.join(
        args.data,
        "multi_checkpoint_{}".format(args.label),
        "ckpt-{}".format(args.ckpt),
    )
    if not os.path.exists(ckpt_path + ".index"):
        print("Checkpoint not found: {}".format(ckpt_path))
        sys.exit(1)

    os.makedirs(args.video_folder, exist_ok=True)
    print("Checkpoint : {}".format(ckpt_path))
    print("Video out  : {}".format(args.video_folder))

    # Config — matches the solve-time architecture
    cfg = ModelCfg()
    cfg.kernel_init_type = "GlorotNormal"
    cfg.layer_sz = [256, 256]

    # Environment chain:
    #   gym(render_mode=rgb_array) → RecordVideo → GymnasiumWrapper → TFPyEnvironment
    # RecordVideo sits below GymnasiumWrapper so its step()/reset() calls are
    # intercepted and frames are captured automatically.
    raw_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    raw_env = _ResizeRenderWrapper(raw_env, args.width, args.height)
    video_env = RecordVideo(
        raw_env,
        video_folder=args.video_folder,
        episode_trigger=lambda _: True,
        name_prefix="{}_ckpt{}".format(args.label, args.ckpt),
        disable_logger=True,
    )
    py_env = GymnasiumWrapper(video_env)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    # Build agent and restore weights
    agent, train_step = _build_agent(tf_env, cfg)
    avg_return_at_save, step = _restore_checkpoint(agent, train_step, ckpt_path)
    print("Restored   : step={:,}, train avg return={:.1f}".format(step, avg_return_at_save))

    # Record episodes
    for ep in range(args.episodes):
        ret, steps = _run_episode(tf_env, agent.policy, max_steps=args.max_steps)
        print("Episode {:>2} : return={:7.1f}, steps={}".format(ep + 1, ret, steps))

    # Close flushes the last video file
    video_env.close()
    tf_env.close()
    print("Done. Videos saved to: {}".format(args.video_folder))


if __name__ == "__main__":
    main()
