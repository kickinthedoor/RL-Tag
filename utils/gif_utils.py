import os
import imageio.v2 as imageio
from PIL import Image
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from config.config import config
from env.tag_env import TagEnv



def generate_gif_from_checkpoint(checkpoint_paths, filename=None, env_name="tag_env", output_dir="gifs", steps_per_game=200):
    """
    Creates a uniquely named checkpoint subdirectory based on the current timestamp.

    Args:
        base_dir (str): Parent directory where checkpoints are stored.

    Returns:
        str: Full path to the newly created checkpoint directory.
    """
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_dir = os.path.join(base_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        for i, checkpoint_path in enumerate(checkpoint_paths):
            print(f"\nLoading checkpoint {i+1}/{len(checkpoint_paths)}:\n{checkpoint_path}")

            trainer = PPOTrainer(config=config)
            trainer.restore(checkpoint_path)

            print("Trainer restored.")

            env = TagEnv()
            env.reset()

            print("Environment created.")

            frames = []

            for _ in range(steps_per_game):
                for agent in env.agent_iter():
                    obs, reward, done, info = env.last()

                    if done:
                        action = None
                    else:
                        action = trainer.compute_action(
                            obs, policy_id=config["multiagent"]["policy_mapping_fn"](agent)
                        )

                    env.step(action)

                    # Convert BytesIO frame to np.array using PIL
                    frame_bytes = env.render_frame()
                    if frame_bytes:
                        frame_image = Image.open(frame_bytes)
                        frames.append(np.array(frame_image))

                    if done:
                        break
            
            print("Game simulated.")

            if frames:
                abs_path = os.path.abspath(os.path.join(output_dir, filename or f"game_{i+1}.gif"))
                print(f"Absolute GIF path: {abs_path}")
                if filename:
                    gif_path = os.path.join(output_dir, filename)
                else:
                    gif_path = os.path.join(output_dir, f"game_4.gif")
                imageio.mimsave(gif_path, frames, duration=0.1)
                print(f"Saved: {gif_path}")
                
            else:
                print("No frames collected.")

            env.close()
            trainer.cleanup()
    except Exception as e:
        print(f"Error during GIF generation: {e}")


def load_snapshot_paths(log_file="snapshot_list.txt"):
    """
    Fetches the checkpoints paths listed in the log_file and returns them.

    Args:
        log_file (str): File containing saved checkpoint paths.

    Returns:
        (list): List of checkpoint paths.
    """
    with open(log_file, "r") as f:
        return [line.strip() for line in f if line.strip()]