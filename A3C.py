import ray
from ray import tune
from ray.rllib.algorithms.a3c import A3C
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import PolicySpec
import wandb
import imageio.v2 as imageio
from env import TagEnv
from rllib_pettingzoo_env import RLlibPettingZooEnv
from customPolicy import CustomA3CPolicyModel
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.models import ModelCatalog

def main():
    wandb.init(
        project="tag-env-project",
        entity="purplewaterfall",
    )

    # Register the custom model
    ModelCatalog.register_custom_model("custom_a3c_policy_model", CustomA3CPolicyModel)

    # Environment creator
    def env_creator(config):
        return TagEnv()

    # RLlib environment config
    rllib_env_config = {"env_creator": env_creator}

    class CustomCallbacks(DefaultCallbacks):
        def __init__(self):
            super().__init__()
            self.total_episodes = 0
            self.total_steps = 0
            self.hunter_wins = 0
            self.prey_wins = 0
            self.warmup_steps = 50000

        def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
            self.total_episodes += 1
            self.total_steps += episode.length

            hunter_info = episode.last_info_for("Hunter")

            hunter_terminated = hunter_info.get("terminated", False)

            if hunter_terminated:  # Hunter wins (caught the prey)
                self.hunter_wins += 1
            else:  # Prey wins (escaped)
                self.prey_wins += 1

            episode.custom_metrics["average_game_length"] = self.total_steps / self.total_episodes
            episode.custom_metrics["hunter_wins"] = self.hunter_wins
            episode.custom_metrics["prey_wins"] = self.prey_wins
            episode.custom_metrics["total_episodes"] = self.total_episodes

    # Class definition
    class CustomRLlibPettingZooEnv(RLlibPettingZooEnv):
        def __init__(self, config):
            super().__init__(config)

    rllib_env = CustomRLlibPettingZooEnv(rllib_env_config)

    # RLlib configuration
    config = {
        "env": CustomRLlibPettingZooEnv,
        "env_config": rllib_env_config,
        "num_workers": 12,
        "num_envs_per_worker":5,  
        "num_gpus": 1, 
        "framework": "torch",
        "entropy_coeff": 0.01,
        "model": {
            "custom_model": "custom_a3c_policy_model",
        },
        "multiagent": {
            "policies": {
                "hunter_policy": PolicySpec(
                    policy_class=None,
                    observation_space=CustomRLlibPettingZooEnv(rllib_env_config).observation_spaces['Hunter'],
                    action_space=CustomRLlibPettingZooEnv(rllib_env_config).action_spaces['Hunter'],
                    config={
                        "model": {
                            "custom_model": "custom_a3c_policy_model",
                        },
                    }
                ),
                "prey_policy": PolicySpec(
                    policy_class=None,
                    observation_space=CustomRLlibPettingZooEnv(rllib_env_config).observation_spaces['Prey'],
                    action_space=CustomRLlibPettingZooEnv(rllib_env_config).action_spaces['Prey'],
                    config={
                        "model": {
                            "custom_model": "custom_a3c_policy_model",
                        },
                    }
                ),
            },
            "policy_mapping_fn": lambda agent_id: "hunter_policy" if agent_id.startswith("Hunter") else "prey_policy",
        },
        "lr_schedule": [
            [0, 1e-4],
            [50000, 1e-4],  # Warmup period
            [500000, 1e-5]  # After warmup
        ],
        "entropy_coeff_schedule": [
            [0, 0.01],
            [50000, 0.01],  # Warmup period
            [500000, 0.001]  # After warmup
        ],
        "rollout_fragment_length": 100,
        "train_batch_size": 500,
        "lr": 1e-4,
        "lambda": 0.95,
        "gamma": 0.99,
        "callbacks": CustomCallbacks,
    }


    # Run training
    analysis = tune.run(
        A3C, 
        config=config,
        stop={"timesteps_total": 100000},
        checkpoint_freq=1, 
        keep_checkpoints_num=5,  
        checkpoint_at_end=True, 
        callbacks=[
            WandbLoggerCallback(
                project="tag-env-project",
                entity="purplewaterfall",
                log_config=True,
            )
        ]
    )

    print("Trial results:")
    for trial in analysis.trials:
        print(f"Trial {trial.trial_id}: {trial.last_result}")

    try:
        best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
        if best_trial:
            best_checkpoint = analysis.get_best_checkpoint(
                trial=best_trial,
                metric="episode_reward_mean",
                mode="max"
            )
            print(f"Best checkpoint path: {best_checkpoint}")
        else:
            best_checkpoint = None
            print("No best trial found.")
    except Exception as e:
        best_checkpoint = None
        print(f"Error retrieving best checkpoint: {e}")

    if best_checkpoint:
        # Restore the trainer with the best checkpoint
        trainer = A3C(config=config)
        trainer.restore(best_checkpoint)

        # Create the environment and capture frames using the trained policy
        env = env_creator(rllib_env_config)
        env.reset()
        frames = []

        for _ in range(200):  # Run for 200 steps to capture the final behavior
            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()  
                if termination or truncation:
                    action = None
                else:
                    action = trainer.compute_action(obs, policy_id=config["multiagent"]["policy_mapping_fn"](agent))
                env.step(action)
                frames.append(env.render_frame())

                if termination or truncation:  
                    break

        if not frames:
            print("No frames captured during the final run.")
        else:
            # Save frames as a GIF file
            video_path = "final_run.gif"
            with imageio.get_writer(video_path, mode='I', duration=0.1) as writer:
                for frame in frames:
                    writer.append_data(imageio.imread(frame))

            # Log the GIF file to wandb
            wandb.log({"final_training_gif": wandb.Video(video_path, format="gif")})
    else:
        print("No valid checkpoint found. Skipping video creation.")    

    # Try to shutdown Ray
    print("Attempting to shutdown Ray...")
    ray.shutdown()
    print("Ray shutdown successfully.")

if __name__ == '__main__':
    main()