import numpy as np

def calculate_distance_with_obstacles(pos1, pos2, obstacle):
    """
    Calculates the Euclidean norm between two positions and adds a penalty, if there is an obstacle between them.

    Args:
        pos1 (np.array): Position of the first entity.
        pos2 (np.array): Position of the second entity.
        obstacle (np.array): Array of coordinates, which the obstacle covers.

    Returns:
        direct_distance (float): The obstacle-penalized distance between two positions.
    """
    # Add a penalty if an obstacle is directly between pos1 and pos2
    direct_distance = np.linalg.norm(pos1 - pos2)
    for ob in obstacle:
        if is_obstacle_between(pos1, pos2, ob):
            return direct_distance + 2.0
    return direct_distance

def is_obstacle_between(pos1, pos2, ob):
    """
    Checks whether an obstacle is located between two positions.

    Args:
        pos1 (np.array): Position of the first entity.
        pos2 (np.array): Position of the second entity.
        ob (np.array): Position of the obstacle.

    Returns:
        (boolean): True, if the given obstacle is between the two positions. False otherwise.
    """
    # Check if the obstacle is in the direct line between pos1 and pos2
    return np.linalg.norm(pos1 - ob) + np.linalg.norm(ob - pos2) <= np.linalg.norm(pos1 - pos2) + 0.1

