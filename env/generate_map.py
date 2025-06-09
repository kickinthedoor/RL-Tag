import numpy as np

def generate_obstacles(grid_size):
    """
    Generates obstacles for the environment. Currently hard coded to return a predetrmined obstacle.

    Args:
        grid_size (tuple): The dimensions for the playing field.

    Returns:
        (np.array): Array of coordinates, which the obstacle covers.
    """
    return np.array([[3, 3], [4, 3], [5, 3], [3, 4], [4, 4], [5, 4], [3, 5], [4, 5], [5, 5]], dtype=np.float32)

def random_position(grid_size, obstacle):
    """
    Used to reposition an agent, when the agent either is on top of the other agent or inside an obstacle during initialization.

    Args:
        grid_size (tuple): The dimensions of the playing field.
        obstacle (np.array): Array of coordinates, which the obstacle covers.

    Returns:
        pos (np.array): The new valid coordinates for the agent.
    """
    while True:
        pos = np.array([np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])], dtype=np.float32)
        if not any(np.array_equal(pos, ob) for ob in obstacle):
            return pos
