from ray.rllib.algorithms.callbacks import DefaultCallbacks
import os

class CustomCallbacks(DefaultCallbacks):
    """
    Custom RLlib callback for tracking episode statistics and saving model snapshots during training.

    This callback:
    - Tracks win rates and average episode lengths.
    - Calculates total and per-step rewards for each agent.
    - Saves model checkpoints at regular training step intervals.
    - Logs the checkpoint paths to a file for later use (e.g., GIF generation).

    Attributes:
        total_episodes (int): Total number of completed episodes.
        total_steps (int): Cumulative number of steps taken across episodes.
        hunter_wins (int): Number of times the hunter has caught the prey.
        prey_wins (int): Number of times the prey has escaped.
        snapshot_interval (int): Number of timesteps between checkpoint snapshots.
        next_snapshot_step (int): Next training step at which to save a checkpoint.
        snapshot_log_file (str): Path to the file where checkpoint paths are recorded.
    """
    def __init__(self):
        """
        Initializes the callback, tracking counters and clearing the snapshot log file.
        """
        super().__init__()
        self.total_episodes = 0
        self.total_steps = 0
        self.snapshot_interval = 100000
        self.next_snapshot_step = self.snapshot_interval
        self.snapshot_log_file = "snapshot_list.txt"

        if os.path.exists(self.snapshot_log_file):
            os.remove(self.snapshot_log_file)

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        """
        Called at the end of each episode.

        Updates win counters, computes average game length, total rewards,
        and reward efficiency per agent. Saves metrics to episode custom metrics.

        Args:
            worker: The rollout worker.
            base_env: The environment instance.
            policies: Dictionary of agent policies.
            episode: The episode object with data for the just-completed episode.
            **kwargs: Additional keyword arguments.
        """
        #print(f"[Episode {episode.episode_id} ended] Agent Infos:")

        hunter_info = None
        prey_info = None

        for agent_id in episode._agent_to_last_info:
            info = episode.last_info_for(agent_id)
            #print(f" - {agent_id}: {info}")

            # Match by name whether tuple or string
            if isinstance(agent_id, tuple):
                name = agent_id[0]
            else:
                name = agent_id

            if name == "Hunter":
                hunter_info = info
            elif name == "Prey":
                prey_info = info

        self.total_episodes += 1
        self.total_steps += episode.length

        hunter_terminated = hunter_info.get("terminated", False) if hunter_info else False

        episode.custom_metrics["hunter_win"] = 1 if hunter_terminated else 0
        episode.custom_metrics["prey_win"] = 1 if not hunter_terminated else 0

        episode.custom_metrics["avg_game_length"] = self.total_steps / self.total_episodes

        # Sum up total episode rewards per agent
        hunter_total_reward = sum(
            reward for (agent, _), reward in episode.agent_rewards.items() if agent == "Hunter"
        )
        prey_total_reward = sum(
            reward for (agent, _), reward in episode.agent_rewards.items() if agent == "Prey"
        )

        episode.custom_metrics["hunter_total_reward"] = hunter_total_reward
        episode.custom_metrics["prey_total_reward"] = prey_total_reward

        # step-by-step reward efficiency
        if episode.length > 0:
            episode.custom_metrics["hunter_reward_per_step"] = hunter_total_reward / episode.length
            episode.custom_metrics["prey_reward_per_step"] = prey_total_reward / episode.length

    def on_episode_step(self, *, episode, **kwargs):
        """
        Called by RLlib at each step of an episode.

        This method ensures that a set of agent IDs is tracked for the episode,
        enabling consistent access to all agents that have provided info dictionaries.
        This is useful for accessing agent-specific info at the end of the episode.

        Args:
            episode: The Episode object containing step-level data.
            **kwargs: Additional arguments passed by RLlib (not used here).
        """
        if not hasattr(episode, "agent_keys"):
            episode.agent_keys = set()
        for agent_id in episode._agent_to_last_info:
            episode.agent_keys.add(agent_id)

    def on_train_result(self, *, trainer, result, **kwargs):
        """
        Called after each training iteration.

        If the total timesteps have reached the defined interval, saves a model checkpoint
        and logs the checkpoint path for future use.

        Args:
            trainer: The RLlib trainer instance.
            result: A dictionary containing training results.
            **kwargs: Additional keyword arguments.
        """
        current_step = result["timesteps_total"]
        if current_step >= self.next_snapshot_step:
            checkpoint_path = trainer.save()

            with open(self.snapshot_log_file, "a") as f:
                f.write(f"{checkpoint_path}\n")

            print(f"[Snapshot] Saved checkpoint at step {current_step}: {checkpoint_path}")
            self.next_snapshot_step += self.snapshot_interval

        # Track learning rate & entropy
        learner_stats = result.get("info", {}).get("learner", {})
        default_policy_stats = learner_stats.get("default_policy", {}).get("learner_stats", {})

        if "cur_lr" in default_policy_stats:
            result["custom_metrics"]["learning_rate"] = default_policy_stats["cur_lr"]

        if "entropy_coeff" in default_policy_stats:
            result["custom_metrics"]["entropy_coeff"] = default_policy_stats["entropy_coeff"]