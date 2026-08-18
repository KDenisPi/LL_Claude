#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_dqn_checkpoint_eli5.py

Feature-importance analysis for the DQN Q-network trained by your
ModelTrain.py / ModelCfg.py, using eli5's permutation importance.

WHY THIS ISN'T JUST A "build_model()" EDIT
--------------------------------------------------------------------------
Your checkpoint is not a bare tf.keras.Model saved as
`tf.train.Checkpoint(model=...)`. Per ModelTrain.init_checkpoints(), it's a
full TF-Agents DQN/DDQN agent, saved as:

    tf.train.Checkpoint(
        step=..., agent=self.agent, policy=self.agent.policy,
        replay_buffer=..., global_step=..., custom_variable=...,
        best_return=...)

TF's object-graph checkpoint format matches variables by walking the
attribute tree starting at each top-level key ("agent", "policy", ...). To
restore correctly we have to reconstruct the *exact same* object graph --
same env specs, same q_network architecture (LYR_0/LYR_1/Output per
ModelCfg.layer_sz), wrapped in the same SelectiveClipDqnAgent -- not a
hand-rolled generic Keras Sequential/Functional model. A plain
`tf.keras.Model` saved under a "model" key would never match an "agent"
key even if the layer shapes happened to line up.

So instead of a standalone build_model(), this script imports your actual
ModelTrain / ModelCfg classes and reuses their real construction logic
(mdl.init_qnet() + mdl.init_agent()) -- this is the only reliable way to
guarantee the object graph matches byte-for-byte what init_checkpoints()
saved.

REQUIREMENTS
--------------------------------------------------------------------------
Run this script from a location where ModelTrain.py, ModelCfg.py,
ModelUtils.py, and gym_wrap.py (all your training-side modules) are
importable -- e.g. drop this file into the same folder as ModelTrain.py,
or add that folder to PYTHONPATH. tensorflow / tf-agents / gymnasium
should already be installed since ModelTrain.py itself depends on them.

    pip install eli5 scikit-learn --break-system-packages

WHAT IT MEASURES
--------------------------------------------------------------------------
eli5.show_weights() on raw Dense-layer weights isn't meaningful here: the
network has two ReLU hidden layers (LYR_0, LYR_1) before the linear Output
layer, so a feature's effect on the chosen action passes through nonlinear
recombination -- reading LYR_0's weights alone tells you nothing about
what action the network ends up picking. Instead this script uses eli5's
permutation importance: it rolls out episodes with the restored greedy
policy to collect (observation, action_taken) pairs, then repeatedly
shuffles one observation feature at a time and measures how often the
network's argmax action changes on the shuffled data vs. the actions it
actually took. A feature the network relies on heavily for action
selection shows a large accuracy drop when shuffled; an unused/irrelevant
feature doesn't.

Usage:
    python analyze_dqn_checkpoint_eli5.py \\
        --checkpoint /path/to/multi_checkpoint_LL_23/ckpt-31 \\
        --data_idx LL_23 \\
        --num_episodes 20 \\
        --output_html importance_report.html

    # If this run used non-default hyperparameters that affect variable
    # SHAPES (hidden layer sizes, kernel init type, env), override them --
    # everything else (learning rate, gamma, epsilon schedule, etc.) does
    # NOT affect variable shapes so it's safe to leave at ModelCfg defaults:
    python analyze_dqn_checkpoint_eli5.py \\
        --checkpoint /path/to/multi_checkpoint_LL_9/ckpt-14 \\
        --data_idx LL_9 \\
        --layer_sz 128 128 \\
        --kernel_init_type GlorotNormal \\
        --env_name LunarLander-v3
"""

import argparse
import sys

import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, ClassifierMixin

import eli5
from eli5.sklearn import PermutationImportance
from eli5.formatters import format_as_text, format_as_html

from ModelCfg import ModelCfg
from ModelTrain import ModelTrain


# Default LunarLander-v3 observation layout (Box(8,)). EDIT this if your
# GymnasiumWrapper transforms, reorders, or extends the raw Gym
# observation -- these labels are only used for display in the report.
DEFAULT_FEATURE_NAMES = [
    "x_pos", "y_pos", "x_vel", "y_vel",
    "angle", "angular_vel", "left_leg_contact", "right_leg_contact",
]


def build_and_restore(cfg: ModelCfg, checkpoint_path: str) -> ModelTrain:
    """
    Recreate the exact env/agent/q_network object graph ModelTrain builds,
    then restore only the `agent` branch of the checkpoint into it. This
    mirrors ModelTrain.evaluate_chkpt(), which deliberately restores just
    {agent, global_step, custom_variable} and skips replay_buffer (its
    capacity at eval time may differ from what was saved, causing a shape
    mismatch) -- here we go even narrower and restore only `agent`, since
    that's the only branch we need for weight/importance analysis.
    """
    mdl = ModelTrain(cfg=cfg)
    mdl.init_qnet()   # builds self.q_net: LYR_0, LYR_1, ..., Output (per ModelCfg.layer_sz)
    mdl.init_agent()  # wraps q_net in SelectiveClipDqnAgent -- same object graph as training

    restore_ckpt = tf.train.Checkpoint(agent=mdl.agent)
    status = restore_ckpt.restore(checkpoint_path)
    # policy/replay_buffer/step/etc. branches are intentionally not requested
    # here, so expect_partial() is required (and expected) rather than a bug.
    status.expect_partial()

    print("Restored agent weights from: {}".format(checkpoint_path))
    return mdl


def collect_observations_and_actions(mdl: ModelTrain, num_episodes: int, max_steps: int = 1000):
    """Roll out the restored greedy policy and record every
    (observation, action_taken) pair seen, across num_episodes episodes."""
    observations, actions = [], []

    for ep in range(num_episodes):
        time_step = mdl._tf_eval_env.reset()
        steps = 0
        while not time_step.is_last() and steps < max_steps:
            obs = time_step.observation.numpy().reshape(-1)
            action_step = mdl.agent.policy.action(time_step)
            action = int(action_step.action.numpy().reshape(-1)[0])

            observations.append(obs)
            actions.append(action)

            time_step = mdl._tf_eval_env.step(action_step.action)
            steps += 1

        print("Episode {}: {} steps".format(ep + 1, steps))

    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.int64)


class QNetworkClassifier(BaseEstimator, ClassifierMixin):
    """Wraps the restored TF-Agents Q-network as an sklearn-style
    multi-class classifier (predicted class = argmax Q-value) so eli5's
    PermutationImportance can treat it like any other black-box estimator."""

    def __init__(self, q_net, num_actions: int):
        self.q_net = q_net
        self.num_actions = num_actions
        self.classes_ = np.arange(num_actions)

    def fit(self, X, y=None):
        return self  # already trained; required by sklearn API (cv="prefit")

    def _q_values(self, X):
        X = tf.convert_to_tensor(np.asarray(X, dtype=np.float32))
        q_values = self.q_net(X, training=False)
        if isinstance(q_values, tuple):  # Sequential network returns (output, network_state)
            q_values = q_values[0]
        return q_values.numpy()

    def predict_proba(self, X):
        q = self._q_values(X)
        q = q - q.max(axis=1, keepdims=True)  # numerically stable softmax
        exp_q = np.exp(q)
        return exp_q / exp_q.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self._q_values(X).argmax(axis=1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Full checkpoint prefix, e.g. /path/to/multi_checkpoint_LL_23/ckpt-31",
    )
    parser.add_argument(
        "--data_idx", required=True,
        help="cfg.data_idx for this run (only used to build layer/init defaults consistently; "
             "does not need to match any file on disk since we bypass checkpoint_dir entirely)",
    )
    parser.add_argument(
        "--layer_sz", nargs="*", type=int, default=None,
        help="Override cfg.layer_sz if this run used non-default hidden layer sizes, e.g. --layer_sz 256 256",
    )
    parser.add_argument(
        "--kernel_init_type", default=None,
        choices=["VarianceScaling", "GlorotNormal", "GlorotUniform"],
        help="Override cfg.kernel_init_type if this run didn't use the ModelCfg default",
    )
    parser.add_argument(
        "--env_name", default=None,
        help="Override cfg.env_name if this run wasn't LunarLander-v3 (affects action/observation spec shapes)",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=20,
        help="Episodes to roll out with the restored greedy policy for collecting (observation, action) pairs",
    )
    parser.add_argument("--n_iter", type=int, default=10, help="Shuffles per feature for permutation importance")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument(
        "--feature_names", nargs="*", default=None,
        help="Override default LunarLander feature names (must match observation dimensionality)",
    )
    parser.add_argument("--output_html", default="importance_report.html")

    args = parser.parse_args()

    cfg = ModelCfg()
    if args.env_name:
        cfg._env_name = args.env_name  # no public setter; env is fixed at __init__ time otherwise
    cfg.data_idx = args.data_idx
    if args.layer_sz:
        cfg.layer_sz = args.layer_sz
    if args.kernel_init_type:
        cfg.kernel_init_type = args.kernel_init_type

    mdl = build_and_restore(cfg, args.checkpoint)

    X, y = collect_observations_and_actions(mdl, args.num_episodes)
    print("Collected {} (observation, action) pairs across {} episodes".format(len(X), args.num_episodes))

    feature_names = args.feature_names or DEFAULT_FEATURE_NAMES
    if len(feature_names) != X.shape[1]:
        print(
            "WARNING: {} feature names given but observations have {} dims -- "
            "falling back to generic names. Pass --feature_names to fix this.".format(
                len(feature_names), X.shape[1]),
            file=sys.stderr,
        )
        feature_names = ["feature_{}".format(i) for i in range(X.shape[1])]

    wrapper = QNetworkClassifier(mdl.q_net, num_actions=mdl._num_actions)

    perm = PermutationImportance(
        wrapper,
        scoring="accuracy",
        n_iter=args.n_iter,
        random_state=args.random_state,
        cv="prefit",  # the network is already trained; don't refit it
    )
    perm.fit(X, y)

    explanation = eli5.explain_weights(perm, feature_names=feature_names)
    print("\n" + format_as_text(explanation))

    html = format_as_html(explanation)
    with open(args.output_html, "w") as f:
        f.write(html)
    print("\nSaved full HTML report to {}".format(args.output_html))


if __name__ == "__main__":
    main()