# RL-Tag
This project is a multi-agent game of catch. The objective is, as traditionally, for the hunter to catch the prey and for the prey to try and escape. 

# Quickstart guide
NOTE! Setting up is required to use GPU, which is needed to run the program.<br/><br/>
Download the repository, optionally create a virtual env using: 'python -m venv venv', activate the venv with: 'venv\Scripts\activate', after which you install the requirements with: 'pip install -r requirements.txt'. Once that is done, you can run the training by simply running 'start.bat' and running 'py scripts\train.py'. 

# Design choices
The algorithm chosen for the model is PPO (Proximal policy optimization), due to its fit for MARL (Multi-agent reinforcement learning), stable training and its fit to both discrete and continuous environments.
Multi-agent tasks introduce an additional layer of interdependencies, where metrics used to analyze single-agent training can't be used the same way. Even more so, when the training consists of a zero-sum game, where  the reward of one agent is mirrored in proportion as a penalty for the other. <br/><br/>
Usual metrics, such as loss, KL divergence, episode reward and policy entropy need to be customized porperly chosen in a way that useful information can be extracted.
Some such metrics could be hunter win rate, prey win rate and average game length. Policy entropy can be used to discern, whether the policy of either agent drops too fast. KL divergence needs to be viewed in relation to each agent, as too high KL indicates instability and near 0 KL indicates stagnation.
The training progress can also be visualized by periodically rendering a simulated game based on the current training progress.

# Ideas and considerations
The map currently is hard coded to be a square grid with a rectangular obstacle in the center. One could improve the generalization ability of the agents by randomly generating a map with obstacles. Training would take longer, of course, but the model itself would learn to generalize better than on a static environment.
Another change that could be made is changing the spawn points of the agents early on to be constant, and later on changing them to random spawns. The idea being that the agents might potentially form strategies early on and later the random spawns could help the generalize. <br/><br/>
As it stands, agents can see the entire environment, including the opponent. A potential change could be to limit the vision to some range. Maybe the hunter and prey could even have unique properties, such as the hunter having narrow, predatory vision, while the prey would have a wider view. The agents could also have different abilities to move. For now both share the same simple actions of moving up, down, left and right.
Next would also be the possibility of testing another algorithm, such as A3C, which would need heavier parameter tuning to properly train the model.


# Training progress
After 2 million iterations, the strategies of the agents look as follows.<br/>
![https://media3.giphy.com/media/aUovxH8Vf9qDu/giphy.gif](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMHF5MXgzbWVwcTFpdWVzMXZjaTljZHV2OXR3aWhrdWVmNmE4ZGc2MiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/GdAX4zqp0lJjdXRb3f/giphy.gif)
<br/><br/>
## Average game length
![avg_game_length](https://github.com/user-attachments/assets/c87f3d23-a3e8-46cc-b46a-3273a27ae16c)
## Policy entropies
![hunter_policy_entropy](https://github.com/user-attachments/assets/a78b69fd-cce7-4b06-aa7b-1f188f02c610)

![prey_policy_entropy](https://github.com/user-attachments/assets/d023811d-dbee-4b1e-be84-59c015ee8ad0)
## Policy KL
![hunter_policy_kl](https://github.com/user-attachments/assets/a4be3193-7b66-4699-a9ff-416105b40631)

![prey_policy_kl](https://github.com/user-attachments/assets/42d11cd2-17b6-4fe3-813a-625b1bec2c50)
## Win rates
![hunter_win_rate](https://github.com/user-attachments/assets/a80785b9-a4e6-4a33-8547-49a8adbeb722)

![prey_win_rate](https://github.com/user-attachments/assets/60e43a3c-861d-4cde-aed9-ddae65a1e0e8)
## Total rewards
![hunter_total_reward](https://github.com/user-attachments/assets/41db0ba1-4d4f-40e2-ac22-8c19ed3e8130)

![prey_total_reward](https://github.com/user-attachments/assets/d26bd846-6347-476e-a1e5-1bcee44b80d5)
## Reward per step
![hunter_reward_per_step](https://github.com/user-attachments/assets/1ccea9ef-6c33-4118-a359-e4127dd01a10)

![prey_reward_per_step](https://github.com/user-attachments/assets/4d9aad44-701c-4c6f-8e4d-43b1b66cb8ea)



# Directories

## 📂 env/
Core environment logic and wrappers (e.g., obstacle handling, observation encoding, and multi-agent interfaces).

```
env/
├── generate_map.py
├── helpers.py
├── observation_encoder.py
├── rllib_wrapper.py
├── tag_env.py
```

## 📂 models/
Additional required models.
```
models/
├── action_mask_model.py
```

## 📂 config/
Holds agent and training configuration logic.

```
config/
├── agent_config.py
├── config.py
```

## 📂 scripts/
Script entry point for training and running the project.

```
scripts/
├── train.py
```

## 📂 utils/
Reusable utilities such as logging, checkpoint management, callbacks, and GIF generation.

```
utils/
├── callbacks.py
├── checkpoint_utils.py
├── gif_utils.py
├── logger.py
```


# Detailed project structure


```
env/
├── generate_map.py
│   ├── generate_obstacles()
│   └── random_position()
├── helpers.py
│   ├── calculate_distance_with_obstacles()
│   └── is_obstacle_between()
├── observation_encoder.py
│   ├── one_hot_encode()
│   └── normalize()
├── tag_env.py
│   └── class TagEnv
│       ├── __init__()
│       ├── step()
│       ├── reset()
│       ├── observe()
│       ├── _was_done_step()
│       ├── get_valid_actions()
│       ├── render()
│       ├── render_frame()
│       ├── close()
│       ├── observation_space()
│       ├── action_space()
│       ├── calculate_rewards()
│       ├── check_terminations()
│       └── check_truncations()

models/
├── action_mask_model.py
│   └── class TorchActionMaskModel
│       ├── __init__()
│       ├── forward()
│       ├── value_function()

config/
├── agent_config.py
│   └── build_multiagent_config()
├── config.py

scripts/
├── train.py
│   └── main()

utils/
├── callbacks.py
│   └── class CustomCallbacks
│       ├── __init__()
│       ├── on_episode_end()
│       └── on_train_result()
├── checkpoint_utils.py
│   ├── create_checkpoint_dir()
│   └── get_best_checkpoint()
├── gif_utils.py
│   ├── generate_gif_from_checkpoint()
│   └── load_snapshot_paths()
└── logger.py
    └── log()
```


