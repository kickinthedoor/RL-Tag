from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym import spaces

class RLlibPettingZooEnv(MultiAgentEnv):
    def __init__(self, config):
        env_creator = config.get("env_creator")
        self.env = env_creator(config)
        self.env.reset()
        self.agents = self.env.agents

        self.observation_spaces = {agent: self.env.observation_spaces[agent] for agent in self.agents}
        self.action_spaces = {agent: self.env.action_spaces[agent] for agent in self.agents}

        self.observation_space = list(self.observation_spaces.values())[0]
        self.action_space = list(self.action_spaces.values())[0]

        print("Initialized RLlibPettingZooEnv")
        print("Observation spaces:", self.observation_spaces)
        print("Action spaces:", self.action_spaces)

    def reset(self):
        self.env.reset()
        observations = {agent: self.env.observe(agent) for agent in self.env.agents}
        return observations

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for agent, action in action_dict.items():
            obs, reward, termination, truncation, info = self.env.step({agent: action})
            observations[agent] = obs
            rewards[agent] = reward
            terminations[agent] = termination
            truncations[agent] = truncation
            infos[agent] = info

        # Add '__all__' to terminations to indicate if the entire environment is done
        dones = {agent: terminations[agent] or truncations[agent] for agent in terminations}
        dones['__all__'] = all(dones.values())

        return observations, rewards, dones, infos

    def render(self, mode="human"):
        return self.env.render(mode)
