import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import sys

from load_dataset import CustomDataset
from snn.cnn_srnn_model import Model
from EnvWrapper_for_trajectory_plot import DroneEnvWrapper


def plot_trajectory_3d(ax, trajectory_coords):
    """
    参数:
    - trajectory_coords: 无人机飞行轨迹坐标的数组，形状为(T, 3)。
    """
    # Translate the trajectory to start at the origin
    trajectory_coords = trajectory_coords - trajectory_coords[0, :]

    # Extract x, y, z coordinates for plotting
    x_coords = trajectory_coords[:, 0]
    y_coords = trajectory_coords[:, 1]
    z_coords = trajectory_coords[:, 2]

    # Plot the 3D trajectory
    ax.plot(x_coords, y_coords, z_coords, color='blue', linewidth=2, linestyle='-')

    # 在起点和终点处绘制特殊标记
    ax.scatter(*trajectory_coords[0, :], color='green', s=90, marker='o', label='Start')
    ax.scatter(*trajectory_coords[-1, :], color='red', s=90, marker='X', label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim((0, 20))
    ax.set_ylim((-10, 10))
    ax.set_zlim((-10, 10))
    ax.set_xticks(np.arange(0, 21, 5))
    ax.set_yticks(np.arange(-10, 11, 5))
    ax.set_zticks(np.arange(-10, 11, 5))
    ax.invert_yaxis()  # 翻转Y轴
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=10)


def fly_trajectory():
    position_list = []
    image_list = []
    salient_map_list = []
    step = 0
    navigation_start_sequence = True
    state, position = env_wrapper.reset()
    salient_map = model.get_salient_map(state, transform=True, combine=False)  # BGR

    position_list.append(position)
    image_list.append(cv2.cvtColor(state, cv2.COLOR_RGB2BGR))  # BGR
    salient_map_list.append(salient_map)  # BGR

    for _ in range(300):  # max time step
        if navigation_start_sequence:
            action = model.predict(state, start_sequence=True)
            navigation_start_sequence = False
        else:
            action = model.predict(state, start_sequence=False)

        next_state, position, done, successful = env_wrapper.step(action)
        salient_map = model.get_salient_map(next_state, transform=True, combine=False)  # BGR

        position_list.append(position)
        image_list.append(cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR))  # BGR
        salient_map_list.append(salient_map)  # BGR

        state = next_state
        step += 1
        if done:
            break

    position_array = np.array(position_list)
    position_array[:, -1] = -position_array[:, -1]  # 反转z轴
    image_array = np.array(image_list)
    salient_map_array = np.array(salient_map_list)

    np.savez('./evaluation/results/noise_analysis/drone_flight_data.npz',
             positions=position_array, images=image_array, salient_maps=salient_map_array)

    # Create a 3D plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    plot_trajectory_3d(ax, position_array)
    plt.show()


if __name__ == '__main__':
    env_wrapper = DroneEnvWrapper(render=True)
    model = Model(load_dataset=False)
    model.load()
    model.register_hook()

    fly_trajectory()
