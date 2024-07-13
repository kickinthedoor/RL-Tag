from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import numpy as np
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import threading
matplotlib.use('Agg') 

x_dim = 8
y_dim = 8


class TagEnv(AECEnv):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.agents = ['Hunter', 'Prey']
        self.agent_selector = agent_selector(self.agents)
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.agents}
        self.max_distance_with_penalty = np.sqrt(x_dim**2 + y_dim**2) + 2.0
        self.obstacle = np.array([[3, 3], [4, 3], [5, 3], [3, 4], [4, 4], [5, 4], [3, 5], [4, 5], [5, 5]], dtype=np.float32)  
        self.obs_shape = x_dim * y_dim * 3 + 1 + 4 # Playing field size thrice for each agent and the obstacles, additionally the normalized distance between agents and low + high dims of borders
        self.observation_spaces = {
            agent: spaces.Dict({
                'observation':spaces.Box(low=0.0, high=1.0,shape=(self.obs_shape,),dtype=np.float32),  
                'action_mask':spaces.Box(low=0, high=1, shape=(self.action_spaces[agent].n,), dtype=np.float32), 
            })
            for agent in self.agents
        }
        self.state = {agent: np.array([5.0, 5.0], dtype=np.float32) for agent in self.agents}
        self.current_agent = None
        self.frames = []
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.agent_selection = self.agent_selector.next()
        self.max_steps = 400
        self.current_step = 0
        self.current_actions = {agent: None for agent in self.agents}
        
    def normalize(self, observation):
        obs_min = np.array([0, 0, 0, 0, 0] + [0, 0] * len(self.obstacle) + [0, 0, 0, 0])
        obs_max = np.array([x_dim, y_dim, x_dim, y_dim, self.max_distance_with_penalty] + [x_dim, y_dim] * len(self.obstacle) + [x_dim, y_dim, x_dim, y_dim])
        normalized_obs = (observation - obs_min) / (obs_max - obs_min)
        return normalized_obs

    def one_hot_encode(self, position, grid_size=8):
        grid_size = int(grid_size)
        one_hot_vector = np.zeros(grid_size * grid_size, dtype=np.float32)
        index = int(position[0]) * grid_size + int(position[1])
        one_hot_vector[index] = 1.0
        return one_hot_vector

    def observe(self, agent):
        other_agent = 'Prey' if agent == 'Hunter' else 'Hunter'
        
        # One-hot encode the agent positions
        one_hot_agent_pos = self.one_hot_encode(self.state[agent], x_dim)
        one_hot_other_agent_pos = self.one_hot_encode(self.state[other_agent], x_dim)
        
        # One-hot encode the obstacle positions
        one_hot_obstacles = np.zeros((x_dim * y_dim,), dtype=np.float32)
        for obstacle in self.obstacle:
            one_hot_obstacles += self.one_hot_encode(obstacle, x_dim)
        
        # Calculate and normalize the distance
        distance = self.calculate_distance_with_obstacles(self.state['Hunter'], self.state['Prey'])
        normalized_distance = distance / self.max_distance_with_penalty  # Normalize to [0, 1]
        
        # Normalize the borders
        normalized_borders = np.array([0 / x_dim, 0 / y_dim, x_dim / x_dim, y_dim / y_dim], dtype=np.float32)

        # Mask actions based on validity
        valid_actions = self.get_valid_actions(agent)
        action_mask = np.zeros(self.action_spaces[agent].n, dtype=np.float32)
        action_mask[valid_actions] = 1.0
        
        # Concatenate all parts of the observation
        observation = np.concatenate((
            one_hot_agent_pos, 
            one_hot_other_agent_pos, 
            [normalized_distance], 
            one_hot_obstacles,
            normalized_borders,
        )).astype(np.float32)
        
        return {'observation':observation,'action_mask':action_mask}

    """ def normalize(self, observation):
        # Example normalization (min-max scaling to [0, 1])
        obs_min = np.array([0, 0, 0, 0, 0] + [0, 0] * len(self.obstacle))
        obs_max = np.array([x_dim, y_dim, x_dim, y_dim, self.max_distance_with_penalty] + [x_dim, y_dim] * len(self.obstacle))
        normalized_obs = (observation - obs_min) / (obs_max - obs_min)
        return normalized_obs """

    def calculate_distance_with_obstacles(self, pos1, pos2):
        # Add a penalty if an obstacle is directly between pos1 and pos2
        direct_distance = np.linalg.norm(pos1 - pos2)
        for ob in self.obstacle:
            if self.is_obstacle_between(pos1, pos2, ob):
                return direct_distance + 2.0
        return direct_distance

    def is_obstacle_between(self, pos1, pos2, ob):
        # Check if the obstacle is in the direct line between pos1 and pos2
        return np.linalg.norm(pos1 - ob) + np.linalg.norm(ob - pos2) <= np.linalg.norm(pos1 - pos2) + 0.1

    def reset(self):
        with self.lock:
            self.state = {
                'Hunter': self._random_position(),
                'Prey': self._random_position()
            }

            # Ensure that the agents do not start at the same position
            while np.array_equal(self.state['Hunter'], self.state['Prey']):
                self.state['Prey'] = self._random_position()

            self.agent_selector.reinit(self.agents)
            self.current_agent = self.agent_selector.next()
            self.agent_selection = self.current_agent
            self.rewards = {agent: 0.0 for agent in self.agents}
            self.truncations = {agent: False for agent in self.agents}
            self.terminations = {agent: False for agent in self.agents}
            self.infos = {agent: {} for agent in self.agents}
            self.frames = []
            self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
            self.current_step = 0
            self.current_actions = {agent: None for agent in self.agents}
            # Capture the initial frame
            self.frames.append(self.render_frame())
            return {agent: self.observe(agent) for agent in self.agents}

    def _was_done_step(self):
        self.current_agent = self.agent_selector.next()
        self.agent_selection = self.current_agent

    def _random_position(self):
        while True:
            pos = np.array([np.random.randint(0, x_dim), np.random.randint(0, y_dim)], dtype=np.float32)
            if not any(np.array_equal(pos, ob) for ob in self.obstacle):
                return pos

    def get_valid_actions(self, agent):
        current_position = self.state[agent]
        valid_actions = []

        # Check each action for validity
        for action in range(4):
            next_position = current_position.copy()
            if action == 0:  # move up
                next_position[1] += 1
            elif action == 1:  # move down
                next_position[1] -= 1
            elif action == 2:  # move left
                next_position[0] -= 1
            elif action == 3:  # move right
                next_position[0] += 1

            # Check if the next position is within bounds
            if 0 <= next_position[0] < x_dim and 0 <= next_position[1] < y_dim:
                # Check if the next position is not an obstacle
                if not any(np.array_equal(next_position, ob) for ob in self.obstacle):
                    valid_actions.append(action)

        return valid_actions

    def step(self, action):
        with self.lock:
            agent = self.current_agent

            if self.terminations[agent] or self.truncations[agent]:
                self._was_done_step()
                return self.observe(agent), self.rewards[agent], self.terminations[agent], self.truncations[agent], {}

            next_position = self.state[agent].copy()
            
            valid_actions = self.get_valid_actions(agent)
            if action not in valid_actions:
                action = np.random.choice(valid_actions)

            if action == 0:  # move up
                next_position[1] += 1
            elif action == 1:  # move down
                next_position[1] -= 1
            elif action == 2:  # move left
                next_position[0] -= 1
            elif action == 3:  # move right
                next_position[0] += 1

            next_position = np.clip(next_position, 0, 8).astype(np.float32)

            #if not any(np.array_equal(next_position, ob) for ob in self.obstacle):
            
            if not (next_position[0] < self.obstacle[:, 0].max() and next_position[0] > self.obstacle[:, 0].min() and next_position[1] < self.obstacle[:, 1].max() and next_position[1] > self.obstacle[:, 1].min()):
                self.state[agent] = next_position

            distance = self.calculate_distance_with_obstacles(self.state['Hunter'], self.state['Prey'])
            if distance < 1.0:
                self.rewards = {'Hunter': 10.0, 'Prey': -10.0}
                self.terminations = {agent: True for agent in self.agents}
                self.infos = {
                    'Hunter': {'terminated': True, 'truncated': False},
                    'Prey': {'terminated': True, 'truncated': False}
                }
            else:
                self.rewards['Hunter'] = 3 - distance - 0.05 * self.current_step
                self.rewards['Prey'] = distance - 3 + 0.05 * self.current_step

            self.current_step += 1
            if self.current_step >= self.max_steps:
                if not self.terminations['Prey'] and not self.terminations['Hunter']:
                    self.rewards['Prey'] += 10.0  
                    self.rewards['Hunter'] -= 10.0

                self.truncations = {agent: True for agent in self.agents}
                self.infos = {
                    'Hunter': {'terminated': False, 'truncated': True},
                    'Prey': {'terminated': False, 'truncated': True}
                }

            termination = self.terminations[agent]
            truncation = self.truncations[agent]
            reward = self.rewards[agent]
            observation = self.observe(agent)
            info = self.infos[agent]
            info['reward'] = reward

            self._cumulative_rewards[agent] += self.rewards[agent]
            self.current_agent = self.agent_selector.next()
            self.agent_selection = self.current_agent


            return observation, reward, termination, truncation, info

    def render_frame(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, x_dim)
        ax.set_ylim(0, y_dim)
        for agent in self.agents:
            ax.plot(self.state[agent][0], self.state[agent][1], 'o', label=agent)
        # Draw the obstacle as a rectangle
        obstacle_x = self.obstacle[:, 0].min()
        obstacle_y = self.obstacle[:, 1].min()
        obstacle_width = self.obstacle[:, 0].max() - obstacle_x 
        obstacle_height = self.obstacle[:, 1].max() - obstacle_y 
        rect = patches.Rectangle((obstacle_x, obstacle_y), obstacle_width, obstacle_height, linewidth=1, edgecolor='red', facecolor='none', label='Obstacle')
        ax.add_patch(rect)
        ax.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf

    def render(self, mode='human'):
        print(f"Hunter: {self.state['Hunter']}, Prey: {self.state['Prey']}")

    def close(self):
        pass

    @property
    def observation_space(self):
        return self.observation_spaces[self.current_agent]

    @property
    def action_space(self):
        return self.action_spaces[self.current_agent]
