import numpy as np
import matplotlib.pyplot as plt
import time
import os

from TD3_Simple_CNN_RNN import TD3
from EnvWrapper_Simple_CNN_RNN_Target_Following import DroneEnvWrapper


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
    ax.set_title('3D Trajectory Visualization')

    ax.legend()
    plt.show()


def plot_trajectories_3d(ax, agent_trajectory, target_trajectory):
    """
    在3D空间中绘制无人机与目标的运动轨迹。
    - agent_trajectory: 无人机飞行轨迹坐标的数组，形状为(T, 3)。
    - target_trajectory: 目标运动轨迹坐标的数组，形状为(T, 3)。
    """
    # Translate the trajectory to start at the origin
    # origin = target_trajectory[0, :] + np.array([10, -10, 0])  # square
    origin = target_trajectory[0, :] + np.array([0, -10, 0])  # circle
    agent_trajectory = agent_trajectory - origin
    target_trajectory = target_trajectory - origin

    # Extract x, y, z coordinates for plotting
    agent_x, agent_y, agent_z = agent_trajectory[:, 0], agent_trajectory[:, 1], agent_trajectory[:, 2]
    target_x, target_y, target_z = target_trajectory[:, 0], target_trajectory[:, 1], target_trajectory[:, 2]

    # Plot the 3D trajectory
    ax.plot(agent_x, agent_y, agent_z, color='blue', linewidth=2, linestyle='-', label='Agent Trajectory')
    ax.plot(target_x, target_y, target_z, color='darkorange', linewidth=2, linestyle='-', label='Target Trajectory')

    # 起点与终点特殊标记
    ax.scatter(*agent_trajectory[0], color='deepskyblue', s=150, marker='o', label='Start')
    ax.scatter(*agent_trajectory[-1], color='green', s=150, marker='*', label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim((-12, 12))
    ax.set_ylim((-12, 12))
    ax.set_zlim((-5, 5))
    # ax.set_xticks(np.arange(0, 16, 5))
    # ax.set_yticks(np.arange(-8, 9, 4))
    # ax.set_zticks(np.arange(-6, 7, 3))
    ax.invert_yaxis()  # 翻转Y轴
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=10)


if __name__ == '__main__':
    r = 10  # radius of the circle
    T = 1600  # total number of time steps
    # trajectory_increments = generate_square_trajectory(r, T, 'bottom_right', clockwise=False)
    trajectory_increments = generate_circular_trajectory(r, T, 'right', clockwise=False)
    trajectory_increments[:, [0, 1]] = trajectory_increments[:, [1, 0]]  # 交换x和y坐标
    # visualize_trajectory(trajectory_increments)

    """==============================Target following=============================="""
    env_wrapper = DroneEnvWrapper(trajectory_increments, render=True)
    td3 = TD3(state_dim=128, action_dim=4)
    td3.load_weights()
    td3.actor.eval()

    agent_position_list = []
    target_position_list = []

    state, agent_position, target_position = env_wrapper.reset()
    agent_position_list.append(agent_position)
    target_position_list.append(target_position)

    for t in range(T):
        action = td3.actor.get_action(state, explore_noise_scale=0)
        next_state, reward, done, agent_position, target_position = env_wrapper.step(action, t)

        agent_position_list.append(agent_position)
        target_position_list.append(target_position)

        state = next_state

    print("Done!")

    agent_position_array = np.array(agent_position_list)
    agent_position_array[:, -1] = -agent_position_array[:, -1]  # 反转z轴
    target_position_array = np.array(target_position_list)
    target_position_array[:, -1] = -target_position_array[:, -1]  # 反转z轴

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_trajectories_3d(ax, agent_position_array, target_position_array)
    plt.tight_layout()
    plt.show()
