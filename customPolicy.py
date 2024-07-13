import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from gym.spaces import Dict as GymDict

torch, nn = try_import_torch()

class CustomA3CPolicyModel(TorchModelV2, nn.Module):
    """PyTorch version of customized attention model with action masking for A3C."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        orig_space = getattr(obs_space, "original_space", obs_space)

        #print("Debug: orig_space.spaces contains:", orig_space.spaces)

        assert isinstance(orig_space, GymDict), f"Expected orig_space to be GymDict but got {type(orig_space)}"
        assert "action_mask" in orig_space.spaces, "'action_mask' key not in original space"
        assert "observation" in orig_space.spaces, "'observation' key not in original space"

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observation"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal"
        )

        self.no_masking = model_config.get("custom_model_config", {}).get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observation"]})

        if self.no_masking:
            return logits, state

        inf_mask = torch.clamp(torch.log(action_mask), min=torch.finfo(torch.float32).min)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


