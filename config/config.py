from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from env.tag_env import TagEnv
from utils.callbacks import CustomCallbacks
from .agent_config import build_multiagent_config


def env_creator(config):
    return TagEnv()

raw_env = TagEnv()
agents = raw_env.agents
observation_spaces = {agent: raw_env.observation_spaces[agent] for agent in agents}
action_spaces = {agent: raw_env.action_spaces[agent] for agent in agents}
multiagent_config = build_multiagent_config(observation_spaces, action_spaces)


config = {
        "env": "tag_env",
        "model": {
            "custom_model": "action_mask_model",
        },
        "num_workers": 4,  #12
        "num_envs_per_worker": 2, #5
        "rollout_fragment_length": 256,
        "num_gpus": 1, 
        "num_gpus_per_worker": 0,
        "framework": "torch",
        "entropy_coeff": 0.01,
        "multiagent": multiagent_config,
        "lr_schedule": [
            [0, 1e-4],
            [50000, 1e-4],  # Warmup period
            [100000, 1e-5]  # After warmup
        ],
        "entropy_coeff_schedule": [
            [0, 0.01],
            [50000, 0.01],  # Warmup period
            [100000, 0.0001]  # After warmup
        ],
        "train_batch_size": 2048,
        "lr": 1e-4,
        "lambda": 0.95,
        "gamma": 0.99,
        "callbacks": CustomCallbacks,
        "sgd_minibatch_size": 512,
        "num_sgd_iter": 20,
        "clip_param": 0.2,
        "grad_clip": 0.5,
    }

eval_config = {
    "env": "tag_env",
    "num_workers": 0,  # Run in main thread
    "framework": "torch",
    "multiagent": multiagent_config,  
    "explore": False,  # Turn off exploration noise
    "render_env": False,  
}