from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override

import torch
import torch.nn as nn

class TorchActionMaskModel(TorchModelV2, nn.Module):
    """
    A custom Torch model for use with RLlib that supports action masking.

    This model uses an internal fully connected network to compute action logits
    and applies a mask to invalidate certain actions based on the action mask
    provided in the observation.

    Observation space must be a Dict with:
        "obs": the actual observation tensor
        "action_mask": a binary vector indicating valid (1) and invalid (0) actions
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.internal_model = FullyConnectedNetwork(
            obs_space.original_space["obs"], action_space, num_outputs, model_config, name + "_fcnet"
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Computes masked action logits from the raw observation and action mask.

        Args:
            input_dict (dict): RLlib input dict with:
                "obs": a dict containing "obs" and "action_mask"
            state (list): RNN hidden states (not used here)
            seq_lens (Tensor): sequence lengths (not used here)

        Returns:
            Tuple:
                masked_logits: logits with -inf added to invalid actions
                state: unchanged RNN state
        """
        obs = input_dict["obs"]["obs"]  # The real observation
        mask = input_dict["obs"]["action_mask"]

        logits, _ = self.internal_model({"obs": obs})
        
        # Mask out invalid actions
        inf_mask = torch.clamp(torch.log(mask), min=torch.finfo(torch.float32).min)
        masked_logits = logits + inf_mask

        return masked_logits, state

    @override(ModelV2)
    def value_function(self):
        """
        Returns the value function output from the internal model.
        Used by RLlib for critic estimation.

        Returns:
            Tensor: value predictions
        """
        return self.internal_model.value_function()