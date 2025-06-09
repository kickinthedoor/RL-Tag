def build_multiagent_config(observation_spaces, action_spaces):
    """
    Builds a multi-agent configuration dictionary for RLlib.
    This configuration defines separate policies for "Hunter" and "Prey" agents,
    using their respective observation and action spaces. A policy mapping function
    is included to assign each agent ID to the appropriate policy.

    Args:
        observation_spaces (dict): A dictionary mapping agent IDs to their observation spaces.
        action_spaces (dict): A dictionary mapping agent IDs to their action spaces.

    Returns:
        (dict): A multi-agent configuration compatible with RLlib's expected format.
    """
    def map_agent(agent_id):
        #print(f"[POLICY MAP] Mapping agent ID: {agent_id}")
        return "hunter_policy" if agent_id == "Hunter" else "prey_policy"
    
    return {
        "policies": {
            "hunter_policy": (None, observation_spaces["Hunter"], action_spaces["Hunter"], {}),
            "prey_policy": (None, observation_spaces["Prey"], action_spaces["Prey"], {}),
        },
        "policy_mapping_fn": map_agent,
    }