import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from utils.checkpoint_utils import create_checkpoint_dir
from utils.logger import log
from utils.checkpoint_utils import get_best_checkpoint
from utils.gif_utils import generate_gif_from_checkpoint,load_snapshot_paths
from config.config import config
import wandb
from ray.air.callbacks.wandb import WandbLoggerCallback
import os
from ray.tune.registry import register_env
from env.tag_env import TagEnv
from ray.rllib.models import ModelCatalog
from models.action_mask_model import TorchActionMaskModel
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv




def main(generate_gif=False):
    """
    Training loop for the game of tag.

    Args:
        generate_gif (boolean): Flag indicating whether a gif should be generated at the end of training to display learned behavior.
    """
    
    wandb.init(
        project="tag-env-project",
        entity="purplewaterfall",
    )
    ray.init()
    register_env("tag_env", lambda config: PettingZooEnv(TagEnv()))
    ModelCatalog.register_custom_model("action_mask_model", TorchActionMaskModel)

    save_dir = create_checkpoint_dir()

    analysis = tune.run(
        PPOTrainer,
        config=config,
        stop={"timesteps_total": 2000000},
        checkpoint_freq=1,
        local_dir=save_dir,
        checkpoint_at_end=True,
        callbacks=[
            WandbLoggerCallback(
                project="tag-env-project",
                entity="purplewaterfall",
                log_config=True,
            ),
        ]
    )

    print("Trial results:")
    for trial in analysis.trials:
        print(f"Trial {trial.trial_id}: {trial.last_result}")

    print("\nGenerating GIFs from saved training snapshots...")

    checkpoints = load_snapshot_paths()
    for cp in checkpoints:
        game_name = os.path.basename(cp)
        generate_gif_from_checkpoint([cp], filename=f"{game_name}.gif")

    best_checkpoint = get_best_checkpoint(analysis)

    if best_checkpoint:
        print(f"Best checkpoint: {best_checkpoint}")
        
        if generate_gif:
            generate_gif_from_checkpoint([best_checkpoint])

    else:
        print("No valid checkpoint found.")

    log(f"Training complete. Results saved at {save_dir}")
    ray.shutdown()

if __name__ == "__main__":
    main(generate_gif=True)