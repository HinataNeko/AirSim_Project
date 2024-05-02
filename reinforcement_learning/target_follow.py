import numpy as np
import matplotlib.pyplot as plt


def generate_circular_trajectory(r, T, start_position='right', clockwise=False):
    """
    Generate increments along a circular trajectory in the xy-plane.

    Parameters:
        r (float): Radius of the circle.
        T (int): Total number of time steps.
        start_position (str): Starting position on the circle ('right', 'top', 'left', 'bottom').
        clockwise (bool): Direction of rotation; clockwise if True.

    Returns:
        numpy.ndarray: An array of shape (T, 3) with increments at each time step in x, y, z coordinates.
    """
    # Map from position description to starting angle
    start_angles = {'right': 0, 'top': np.pi / 2, 'left': np.pi, 'bottom': 3 * np.pi / 2}

    # Determine the start angle based on the starting position
    start_angle = start_angles[start_position]

    # Determine the direction of the angle increment
    angle_increment = 2 * np.pi / T
    if clockwise:
        angle_increment = -angle_increment

    # Array to hold the increments in the coordinates at each time step
    increments = np.zeros((T, 3))

    # Generate the increments for each time step
    for t in range(T):
        # Calculate the angle for the current time step
        angle = start_angle + t * angle_increment

        # Calculate the x and y increments using the cosine and sine of the angle
        x_increment = r * np.cos(angle) - r * np.cos(angle - angle_increment)
        y_increment = r * np.sin(angle) - r * np.sin(angle - angle_increment)

        # z_increment is zero since the circle is in the xy-plane
        z_increment = 0

        # Store the increments
        increments[t] = [x_increment, y_increment, z_increment]

    return increments


def generate_square_trajectory(r, T, start_corner='top_right', clockwise=False):
    """
    Generate increments along a square trajectory.

    Parameters:
        r (float): Half the length of the side of the square.
        T (int): Total number of time steps.
        start_corner (str): Starting corner on the square ('top_right', 'top_left', 'bottom_left', 'bottom_right').
        clockwise (bool): Direction of rotation; clockwise if True.

    Returns:
        numpy.ndarray: An array of shape (T, 2) with increments at each time step in x, y coordinates.
    """

    # Map from corner description to initial index
    corners = {'top_right': 0, 'top_left': 1, 'bottom_left': 2, 'bottom_right': 3}

    # Coordinate sequence for corners depending on clockwise or counterclockwise movement
    if clockwise:
        corner_sequence = np.array([[r, r], [r, -r], [-r, -r], [-r, r]])
    else:
        corner_sequence = np.array([[r, r], [-r, r], [-r, -r], [r, -r]])

    # Starting index for the corner sequence
    start_index = corners[start_corner]
    if clockwise:
        ordered_corners = np.roll(corner_sequence, start_index, axis=0)
    else:
        ordered_corners = np.roll(corner_sequence, -start_index, axis=0)

    # Number of time steps per side
    steps_per_side = T // 4

    # Array to hold the increments in the coordinates at each time step
    increments = np.zeros((T, 3))

    # Generate the increments for each side of the square
    for i in range(4):
        start_corner = ordered_corners[i]
        end_corner = ordered_corners[(i + 1) % 4]

        # Linear interpolation between corners
        for t in range(steps_per_side):
            increments[i * steps_per_side + t, :2] = (end_corner - start_corner) / steps_per_side

    return increments


def visualize_trajectory(increments):
    """
    Visualize the trajectory in 3D space given the increments.

    Parameters:
        increments (numpy.ndarray): An array of shape (T, 3) with increments at each time step in x, y, z coordinates.
    """
    # Calculate the cumulative sum of increments to get the positions
    positions = np.cumsum(increments, axis=0)

    # Create a new figure for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Trajectory')

    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Circular Trajectory Visualization')

    ax.legend()
    plt.show()


if __name__ == '__main__':
    r = 5  # radius of the circle
    T = 1000  # total number of time steps
    start_position = 'top'  # start from the top of the circle
    clockwise = True  # move in a clockwise direction
    trajectory_increments = generate_circular_trajectory(r, T, start_position, clockwise)
    visualize_trajectory(trajectory_increments)
