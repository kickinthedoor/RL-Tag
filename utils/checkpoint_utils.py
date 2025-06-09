import os
from datetime import datetime

def create_checkpoint_dir(base_dir="checkpoints"):
    """
    Creates a uniquely named checkpoint subdirectory based on the current timestamp.

    Args:
        base_dir (str): Parent directory where checkpoints are stored.

    Returns:
        (str): Full path to the newly created checkpoint directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create new directory path
    checkpoint_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    return checkpoint_dir


def get_best_checkpoint(analysis, metric="episode_reward_mean"):
    """
    Fetches the best checkpoint path based on a chosen metric, when given an analysis.

    Args:
        analysis (ExperimentAnalysis object): Object containing information on all trials during training.
        metric (str): The metric based on which the best checkpoint will be determined.

    Returns:
        best_checkpoint (str): Path to the best checkpoint, if exists. Otherwise None.
    """
    try:
        best_trial = analysis.get_best_trial(metric, mode="max")
        if best_trial:
            best_checkpoint = analysis.get_best_checkpoint(
                trial=best_trial,
                metric=metric,
                mode="max"
            )
            return best_checkpoint
        else:
            return None
    except Exception as e:
        print(f"Error retrieving best checkpoint: {e}")
        return None