import numpy as np

def one_hot_encode(position, grid_size):
    """
    One hot encoding a position on the playing field. In essence, one hot encoding fills a matrix with 0s and only 
    the positions matching certain criteria (like the position for us) are identified by 1s.

    Args:
        position (np.array): Position of some entity on the playing field.
        grid_size (tuple): The dimensions for the playing field.
    
    Returns:
        vec (np.array): One hot encoded position on the playing field.
    """
    vec = np.zeros(grid_size[0] * grid_size[1], dtype=np.float32)
    idx = int(position[0]) * grid_size[1] + int(position[1])
    vec[idx] = 1.0
    return vec

#unused
def normalize(self, observation,x_dim,y_dim):
    obs_min = np.array([0, 0, 0, 0, 0] + [0, 0] * len(self.obstacle) + [0, 0, 0, 0])
    obs_max = np.array([x_dim, y_dim, x_dim, y_dim, self.max_distance_with_penalty] + [x_dim, y_dim] * len(self.obstacle) + [x_dim, y_dim, x_dim, y_dim])
    normalized_obs = (observation - obs_min) / (obs_max - obs_min)
    return normalized_obs
