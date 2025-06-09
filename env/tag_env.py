from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gym import spaces
import threading
import numpy as np
from .generate_map import generate_obstacles, random_position
from .observation_encoder import one_hot_encode
from .helpers import calculate_distance_with_obstacles
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class TagEnv(AECEnv):
    """
     A custom multi-agent environment for a grid-based tag game using PettingZoo's AEC interface.

    Agents:
        - 'Hunter': attempts to catch the prey.
        - 'Prey': attempts to escape the hunter.

    The environment includes movement constraints, obstacle avoidance, and strategic reward shaping
    to simulate pursuit-evasion behavior.

    Attributes:
        grid_size (tuple): Dimensions of the grid (x_dim, y_dim).
        max_steps (int): Maximum number of environment steps per episode.
        lock (threading.Lock): Thread-safety for asynchronous use cases.
        agents (list): List of agent names.
        action_spaces (dict): Mapping from agent to its action space.
        observation_spaces (dict): Mapping from agent to its observation space.
        max_distance_with_penalty (float): Max distance between agents + penalty if there is an obstacle inbetween.
        state (dict): Current position of each agent.
        rewards, terminations, truncations, infos, etc.: RL signals for PettingZoo compliance.
    """

    def __init__(self, grid_size=(8, 8), max_steps=400):
        """
        Initialize the tag environment.

        Args:
            grid_size (tuple): Width and height of the grid.
            max_steps (int): Maximum number of steps in an episode before truncation.
        """
        super().__init__()
        self.grid_size = grid_size
        self.x_dim = grid_size[0]
        self.y_dim = grid_size[1]
        self.max_steps = max_steps
        self.lock = threading.Lock()

        self.agents = ['Hunter', 'Prey']
        self.agent_selector = agent_selector(self.agents)
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.agents}

        self.obstacle = generate_obstacles(grid_size)
        self.max_distance_with_penalty = np.sqrt(self.x_dim**2 + self.y_dim**2) + 2.0

        obs_shape = self.x_dim * self.y_dim * 3 + 1 + 4
        self.observation_spaces = {
            agent: spaces.Dict({
                'obs': spaces.Box(low=0.0, high=1.0, shape=(obs_shape,), dtype=np.float32),
                'action_mask': spaces.Box(low=0, high=1, shape=(self.action_spaces[agent].n,), dtype=np.float32),
            })
            for agent in self.agents
        }

        self.reset()

    def observe(self, agent):
        """
        Return the current observation for the specified agent.

        Args:
            agent (str): Agent ID ('Hunter' or 'Prey').

        Returns:
            dict: A dictionary containing the observation vector and a binary action mask.
        """
        other_agent = 'Prey' if agent == 'Hunter' else 'Hunter'
        
        # One-hot encode the agent positions
        one_hot_agent_pos = one_hot_encode(self.state[agent], [self.x_dim,self.y_dim])
        one_hot_other_agent_pos = one_hot_encode(self.state[other_agent], [self.x_dim,self.y_dim])
        
        # One-hot encode the obstacle positions
        one_hot_obstacles = np.zeros((self.x_dim * self.y_dim,), dtype=np.float32)
        for obstacle in self.obstacle:
            one_hot_obstacles += one_hot_encode(obstacle, [self.x_dim,self.y_dim])
        
        # Calculate and normalize the distance
        distance = calculate_distance_with_obstacles(self.state['Hunter'], self.state['Prey'],self.obstacle)
        normalized_distance = distance / self.max_distance_with_penalty  # Normalize to [0, 1]
        
        # Normalize the borders
        normalized_borders = np.array([0 / self.x_dim, 0 / self.y_dim, self.x_dim / self.x_dim, self.y_dim / self.y_dim], dtype=np.float32)

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

        if np.any(np.isnan(observation)):
            raise ValueError(f"NaN detected in observation for agent {agent} at step {self.current_step}")
        if np.any(np.isinf(observation)):
            raise ValueError(f"Inf detected in observation for agent {agent} at step {self.current_step}")
        
        #print(f"[ENV] Observation for {agent}:")
        #print(f" - obs shape: {observation.shape}, NaNs: {np.isnan(observation).any()}")
        #print(f" - mask shape: {action_mask.shape}, values: {action_mask}")
        
        return {'obs':observation,'action_mask':action_mask}

    def reset(self):
        """
        Resets the environment state and agent positions.

        Returns:
            dict: Initial observations for all agents.
        """
        with self.lock:
            self.agents = ['Hunter', 'Prey']
            self.state = {
                'Hunter': random_position(self.grid_size, self.obstacle),
                'Prey': random_position(self.grid_size, self.obstacle),
            }
            while np.array_equal(self.state['Hunter'], self.state['Prey']):
                self.state['Prey'] = random_position(self.grid_size, self.obstacle)

            self.agent_selector.reinit(self.agents)
            self.agent_selection = self.agent_selector.next()

            self.rewards = {agent: 0.0 for agent in self.agents}
            self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
            self.terminations = {agent: False for agent in self.agents}
            self.truncations = {agent: False for agent in self.agents}
            self.infos = {agent: {} for agent in self.agents}
            self.current_step = 0

            return {agent: self.observe(agent) for agent in self.agents}

    def step(self, action):
        """
        Performs one environment step for the current agent.

        Args:
            action (int): The chosen discrete action.

        Returns:
            Tuple containing (observation, reward, termination, truncation, info)
        """
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_done_step()
            return

        pos = self.state[agent].copy()
        move = [(0, 1), (0, -1), (-1, 0), (1, 0)][action]
        new_pos = np.clip(pos + move, [0, 0], np.array(self.grid_size) - 1)

        if not any(np.array_equal(new_pos, ob) for ob in self.obstacle):
            self.state[agent] = new_pos

        self.current_step += 1

        distance = calculate_distance_with_obstacles(self.state['Hunter'], self.state['Prey'],self.obstacle)

        self.calculate_rewards(distance)
        self.check_terminations(distance)
        self.check_truncations()
        

        if np.isnan(self.rewards[agent]) or np.isinf(self.rewards[agent]):
            raise ValueError(f"Invalid reward ({self.rewards[agent]}) for agent {agent} at step {self.current_step}")

        self._cumulative_rewards[agent] += self.rewards[agent]

        for agent in self.agents:
            self.infos[agent] = {
                "terminated": self.terminations[agent],
                "truncated": self.truncations[agent],
                "step": self.current_step,
            }

        self.agent_selection = self.agent_selector.next()

        if all(self.terminations[a] or self.truncations[a] for a in self.agents):
            #print(f"[END DETECTED] Step: {self.current_step}, Agents done. Reset soon.")
            self._episode_ended = True

    
    def calculate_rewards(self,distance):
        """
        Compute the current reward based on agent distance.

        Args:
            distance (float): Distance between hunter and prey.
        """
        if distance < 1.0:
            self.rewards = {'Hunter': 10.0, 'Prey': -10.0}
        else:
            self.rewards['Hunter'] = 3 - distance - 0.05 * self.current_step
            self.rewards['Prey'] = distance - 3 + 0.05 * self.current_step

    def check_terminations(self,distance):
        """
        Set termination flags if the hunter catches the prey.

        Args:
            distance (float): Distance between hunter and prey.
        """
        if distance < 1.0:
            print("THE PREY HAS BEEN CAUGHT!!!")
            self.terminations = {agent: True for agent in self.agents}
            for agent in self.agents:
                self.infos[agent] = {
                    'terminated': True,
                    'truncated': False,
                    'step': self.current_step
                }
            #print(f"[TagEnv] Termination triggered! Terminations: {self.terminations}")

    
    def check_truncations(self):
        """
        Set truncation flags if max steps are reached.
        """
        if self.current_step >= self.max_steps:
            if not self.terminations['Prey'] and not self.terminations['Hunter']:
                print("THE PREY HAS ESCAPED!!!")
                self.rewards['Prey'] += 10.0
                self.rewards['Hunter'] -= 10.0

            self.truncations = {agent: True for agent in self.agents}
            for agent in self.agents:
                self.infos[agent] = {
                    'terminated': False,
                    'truncated': True,
                    'step': self.current_step
                }
            
    def get_valid_actions(self, agent):
        """
        Return a list of valid actions for the given agent.

        Args:
            agent (str): Agent name.

        Returns:
            valid_actions (list): Indices of legal actions.
        """
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
            if 0 <= next_position[0] < self.x_dim and 0 <= next_position[1] < self.y_dim:
                # Check if the next position is not an obstacle
                if not any(np.array_equal(next_position, ob) for ob in self.obstacle):
                    valid_actions.append(action)

        return valid_actions

    def _was_done_step(self):
        """
        Skip agent if it was already terminated or truncated.
        """
        agent = self.agent_selection
        obs = self.observe(agent)
        self._cumulative_rewards[agent] += 0.0
        self.agent_selection = self.agent_selector.next()

        # Delay agent clearing until final 'last()' is called
        if hasattr(self, "_episode_ended") and self._episode_ended:
            self.agents = []
            self._episode_ended = False

    def render(self, mode='human'):
        """
        Print the current state of the agents.
        """
        print(f"Hunter: {self.state['Hunter']}, Prey: {self.state['Prey']}")

    def render_frame(self):
        """
        Generate a single frame of the environment as a PNG image.

        Returns:
            buf (BytesIO): Buffer containing PNG image of current state.
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.x_dim)
        ax.set_ylim(0, self.y_dim)
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
    
    def close(self):
        """
        Close the environment (no-op placeholder).
        """
        pass


    def observation_space(self,agent):
        """
        Get the current agent's observation space.

        Required by RLlib to infer observation space dynamically during training.
        """
        return self.observation_spaces[agent]


    def action_space(self,agent):
        """
        Get the current agent's action space.

        Required by RLlib to infer action space dynamically during training.
        """
        return self.action_spaces[agent]
    
    def last(self):
        """
        Returns the most recent observation, reward, done status, and info for the current agent.

        This method is used by PettingZoo environments to provide turn-based data to the agent
        whose turn it is (i.e., the current 'agent_selection'). Expected to be called before
        deciding on the next action for the agent.

        Returns:
            tuple:
                obs (dict): The latest observation for the current agent.
                reward (float): The reward received by the current agent from the last step.
                done (bool): Whether the current agent's episode has terminated or truncated.
                info (dict): Additional information associated with the current agent's state.
        """
        agent = self.agent_selection
        obs = self.observe(agent)
        reward = self.rewards[agent]
        done = self.terminations[agent] or self.truncations[agent] 
        info = self.infos.get(agent, {})
        #print(f"[LAST] Agent: {agent} | Done: {done} | Info: {info}")
        return obs, reward, done, info  
    
    @property
    def dones(self):
        """
        Returns the 'done' status for all agents in the environment.

        Each agent is marked as done if either it has terminated (due to a catching event)
        or the episode has been truncated (reaching the maximum step limit).

        Returns:
            dict: A dictionary mapping each agent's name to a boolean done flag.
        """
        return {
            agent: self.terminations[agent] or self.truncations[agent]
            for agent in self.agents
        }
