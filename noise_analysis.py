import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import sys
import os

from load_dataset import CustomDataset
from evaluation.cnn_rnn_model import Model
# from evaluation.cnn_lstm_model import Model
# from evaluation.cnn_gru_model import Model
# from snn.cnn_srnn_model import Model
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
    ax.plot(x_coords, y_coords, z_coords, color='blue', linewidth=2, linestyle='-', label='trajectory')

    # 在起点和终点处绘制特殊标记
    ax.scatter(*trajectory_coords[0], color='deepskyblue', s=90, marker='o', label='Start')
    ax.scatter(*trajectory_coords[-1], color='green', s=90, marker='*', label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim((0, 15))
    ax.set_ylim((-8, 8))
    ax.set_zlim((-6, 6))
    ax.set_xticks(np.arange(0, 16, 5))
    ax.set_yticks(np.arange(-8, 9, 4))
    ax.set_zticks(np.arange(-6, 7, 3))
    ax.invert_yaxis()  # 翻转Y轴
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=10)


def plot_2_trajectories_3d(ax, trajectory_coords1, trajectory_coords2, successful1, successful2):
    """
    在3D空间中绘制两条无人机飞行轨迹，并根据轨迹的导航成功与否在终点使用不同的标记。
    - trajectory_coords: 无人机飞行轨迹坐标的数组，形状为(T, 3)。
    """
    # Translate the trajectory to start at the origin
    trajectory_coords1 = trajectory_coords1 - trajectory_coords1[0, :]
    trajectory_coords2 = trajectory_coords2 - trajectory_coords2[0, :]

    # Extract x, y, z coordinates for plotting
    x_coords1, y_coords1, z_coords1 = trajectory_coords1[:, 0], trajectory_coords1[:, 1], trajectory_coords1[:, 2]
    x_coords2, y_coords2, z_coords2 = trajectory_coords2[:, 0], trajectory_coords2[:, 1], trajectory_coords2[:, 2]

    # Plot the 3D trajectory
    ax.plot(x_coords1, y_coords1, z_coords1, color='blue', linewidth=2, linestyle='-', label='Without Noise')
    ax.plot(x_coords2, y_coords2, z_coords2, color='darkorange', linewidth=2, linestyle='-', label='With Noise')

    # 起点特殊标记
    ax.scatter(*trajectory_coords1[0], color='deepskyblue', s=90, marker='o', label='Start')

    # 根据导航成功与否在终点绘制不同的标记
    end_marker1 = ('*', 'green') if successful1 else ('X', 'red')
    end_marker2 = ('*', 'green') if successful2 else ('X', 'red')
    ax.scatter(*trajectory_coords1[-1], color=end_marker1[1], s=90, marker=end_marker1[0],
               label='Successful' if successful1 else 'Failed')
    ax.scatter(*trajectory_coords2[-1], color=end_marker2[1], s=90, marker=end_marker2[0],
               label='Successful' if successful2 else 'Failed')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim((0, 15))
    ax.set_ylim((-8, 8))
    ax.set_zlim((-6, 6))
    ax.set_xticks(np.arange(0, 16, 5))
    ax.set_yticks(np.arange(-8, 9, 4))
    ax.set_zticks(np.arange(-6, 7, 3))
    ax.invert_yaxis()  # 翻转Y轴
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=10)


def plot_n_trajectories_3d(ax, trajectory_coords_list, success_list, label_list, color_list=None):
    """
    在3D空间中绘制n条无人机飞行轨迹，并根据每条轨迹的导航成功与否在终点使用不同的标记。
    - ax: 3D坐标轴。
    - trajectory_coords_list: 各条轨迹坐标的列表，每个元素形状为 (T, 3)。
    - success_list: 每条轨迹是否成功到达的布尔列表。
    - labels_list: 每条轨迹的描述性标签列表。
    - color_list: （可选）轨迹颜色的列表，如果未提供，则自动生成颜色。
    """
    if color_list is None:
        # 如果未提供颜色列表，生成默认颜色
        color_list = plt.cm.viridis(np.linspace(0, 1, len(trajectory_coords_list)))

    for idx, (trajectory_coords, successful, label) in enumerate(
            zip(trajectory_coords_list, success_list, label_list)):
        trajectory_coords = trajectory_coords - trajectory_coords[0, :]  # 平移轨迹起始至原点
        x, y, z = trajectory_coords[:, 0], trajectory_coords[:, 1], trajectory_coords[:, 2]

        ax.plot(x, y, z, color=color_list[idx], linewidth=2, linestyle='-', label=label)

        # 特殊标记起点
        if idx == 0:
            ax.scatter(*trajectory_coords[0], color='deepskyblue', s=90, marker='o', label='Start')

        # 根据导航成功与否在终点绘制不同的标记
        end_marker = ('*', 'green') if successful else ('X', 'red')
        ax.scatter(*trajectory_coords[-1], color=end_marker[1], s=90, marker=end_marker[0],
                   label='Successful' if successful else 'Failed')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, 15])
    ax.set_ylim([-8, 8])
    ax.set_zlim([-6, 6])
    ax.set_xticks(np.arange(0, 16, 5))
    ax.set_yticks(np.arange(-8, 9, 4))
    ax.set_zticks(np.arange(-6, 7, 3))
    ax.invert_yaxis()  # 翻转Y轴
    ax.legend()


def fly_trajectory():
    position_list = []
    image_list = []
    salient_map_list = []
    step = 0
    navigation_start_sequence = True
    state, position = env_wrapper.reset()
    salient_map = model.get_salient_map(state, transform=True, combine=False)  # BGR, np.uint8, (H, W, C)

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
        salient_map = model.get_salient_map(next_state, transform=True, combine=False)  # BGR, np.uint8, (H, W, C)

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

    np.savez(record_path, positions=position_array, images=image_array, salient_maps=salient_map_array)
    print(f"Record saved to: {record_path}")

    # Create a 3D plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    plot_trajectory_3d(ax, position_array)
    plt.tight_layout()
    plt.show()


def play_record_as_video(image_array, salient_map_array, start_frame=0):
    """
    :param image_array: np.uint8, (N, H, W, C), BGR format
    :param salient_map_array: np.uint8, (N, H, W, C), BGR format
    """
    frames = image_array.shape[0]
    idx = start_frame
    cv2.destroyAllWindows()
    while 0 <= idx < frames:
        # Combining image and salient map vertically
        combined_frame = np.vstack((image_array[idx], salient_map_array[idx]))
        combined_frame = cv2.putText(combined_frame, f'Frame: {idx}', (10, 40),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Video', combined_frame)

        key = cv2.waitKey(0)
        if key == 32:  # Space bar for next frame
            idx += 1
            idx = min(idx, frames - 1)
        elif key == 110:  # 'n' key for skipping 10 frames
            idx += 10
            idx = min(idx, frames - 1)
        elif key in [98, 66]:  # 'b' or 'B' key for going back one frame
            idx = max(0, idx - 1)
        elif key == 99 or key == 67:  # 'c' or 'C' key to exit
            break
        elif key in [115, 83]:  # 's' or 'S' key to save current frame's image and salient map
            image_filename = os.path.join(save_path, 'images', f'{name}_{mode}_image_{idx}.png')
            salient_map_filename = os.path.join(save_path, 'images', f'{name}_{mode}_salient_map_{idx}.png')
            cv2.imwrite(image_filename, image_array[idx])
            cv2.imwrite(salient_map_filename, salient_map_array[idx])
            print(f'Saved image and salient map for frame {idx}.')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    env_wrapper = DroneEnvWrapper(render=True, image_noise=True)
    model = Model(load_dataset=False)
    model.load()
    model.register_hook()

    name = 'gru'
    mode = 'noise_0.3'
    save_path = './evaluation/results/noise_analysis'
    record_path = os.path.join(save_path, f'{name}_{mode}.npz')

    # Fly trajectory
    # if os.path.isfile(record_path):
    #     print("\033[91m\033[1mWarning, the path is already existed!\033[0m")
    # fly_trajectory()

    # Play record as video
    record_data = np.load(record_path)
    image_array = record_data['images']
    salient_map_array = record_data['salient_maps']
    play_record_as_video(image_array, salient_map_array, start_frame=0)

    # plot 2 trajectories
    # record_path1 = os.path.join(save_path, f'{name}_no_noise.npz')
    # record_path2 = os.path.join(save_path, f'{name}_with_noise.npz')
    # record_data1 = np.load(record_path1)
    # record_data2 = np.load(record_path2)
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111, projection='3d')
    # plot_2_trajectories_3d(ax, record_data1['positions'], record_data2['positions'],
    #                        True, True)
    # plt.tight_layout()
    # plt.show()

    # plot n trajectories
    # record_path1 = os.path.join(save_path, f'{name}_no_noise.npz')
    # record_path2 = os.path.join(save_path, f'{name}_noise_0.3.npz')
    # record_path3 = os.path.join(save_path, f'{name}_noise_0.5.npz')
    # record_data1 = np.load(record_path1)
    # record_data2 = np.load(record_path2)
    # record_data3 = np.load(record_path3)
    # trajectories = [record_data1['positions'], record_data2['positions'], record_data3['positions']]
    # successes = [True, True, True]
    # labels = ["No Noise", "Gaussian Noise, Variance = 0.3", "Gaussian Noise, Variance = 0.5"]
    # colors = ['blue', 'limegreen', 'darkorange']
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111, projection='3d')
    # plot_n_trajectories_3d(ax, trajectories, successes, labels, colors)
    # plt.tight_layout()
    # plt.show()
