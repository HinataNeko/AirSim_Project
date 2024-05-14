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

from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer, monitor

from load_dataset import CustomDataset
from cnn_srnn_model import Model
from EnvWrapper_for_trajectory_plot import DroneEnvWrapper

all_dataset = CustomDataset("./datasets/uav_recording", enhance=False, random_clip=False)
all_dataloader = DataLoader(all_dataset, batch_size=1, shuffle=False)

model = Model()
model.load()
net = model.model
net.eval()

# 第一层的脉冲发放率和第二层的最后时刻膜电位
all_input = []
all_spike_rate1 = []
all_v_last1 = []
all_v_last2 = []
all_output = []

# 神经元构造参数中的 store_v_seq: bool = False 表示在默认情况下，只记录当前时刻的电压，不记录所有时刻的电压序列
# 现在我们想记录所有时刻的电压，则将其更改为 True
for m in net.modules():
    if isinstance(m, neuron.LIFNode):
        m.store_v_seq = True

# SpikingRNN第一层
v_seq_monitor1 = monitor.AttributeMonitor(attribute_name='v_seq', pre_forward=False,
                                          net=net.srnn.rnn1, instance=neuron.LIFNode)
spike_seq_monitor1 = monitor.OutputMonitor(net=net.srnn.rnn1, instance=neuron.LIFNode)

# SpikingRNN第二层
v_seq_monitor2 = monitor.AttributeMonitor(attribute_name='v_seq', pre_forward=False,
                                          net=net.srnn.rnn2, instance=neuron.LIFNode)

# with torch.no_grad():
#     for batch_data, _ in tqdm(all_dataloader, file=sys.stdout, leave=False):
#         all_input.append(batch_data.squeeze(0))
#         batch_data = batch_data.to(model.device)
#         batch_output = net(batch_data)  # (batch_size, frames, output)
#
#         all_spike_rate1.extend(spike_seq_monitor1.records)
#         all_v_last1.extend(v_seq_monitor1.records)
#         all_v_last2.extend(v_seq_monitor2.records)
#         all_output.append(batch_output.squeeze(0))
#
#         spike_seq_monitor1.clear_recorded_data()  # list: [Tensor: (T, batch_size, hidden)]
#         v_seq_monitor1.clear_recorded_data()  # list: [Tensor: (T, batch_size, hidden)]
#         v_seq_monitor2.clear_recorded_data()  # list: [Tensor: (T, batch_size, hidden)]
#
#     all_spike_rate1 = torch.stack(all_spike_rate1, dim=1).squeeze(2)  # (T, N, hidden), N为所有图像数
#     all_v_last1 = torch.stack(all_v_last1, dim=1).squeeze(2)  # (T, N, hidden), N为所有图像数
#     all_v_last2 = torch.stack(all_v_last2, dim=1).squeeze(2)  # (T, N, hidden), N为所有图像数
#
#     all_T_spikes1 = all_spike_rate1.cpu().numpy()  # (T, N, hidden), N为所有图像数
#     all_T_v1 = all_v_last1.cpu().numpy()  # (T, N, hidden), N为所有图像数
#     all_T_v2 = all_v_last2.cpu().numpy()  # (T, N, hidden), N为所有图像数
#     all_spike_rate1 = torch.mean(all_spike_rate1, dim=0).cpu().numpy()  # (N, hidden), N为所有图像数
#     all_v_last2 = all_v_last2[-1, ...].cpu().numpy()  # (N, hidden), N为所有图像数
#
#     all_input = torch.cat(all_input, dim=0).cpu().numpy()  # (N, C, H, W), N为所有图像数
#     all_output = torch.cat(all_output, dim=0).cpu().numpy()  # (N, output), N为所有图像数

# n_hidden = 128
# n_output = 4

# all_input: (N, C, H, W)
# all_output: (N, output)
# all_T_spikes1: (T, N, hidden)
# all_T_v1: (T, N, hidden)
# all_T_v2: (T, N, hidden)
# all_v_last2: (N, hidden)


"""==============================SpikingRNN第一层，图像位置对应的神经元数量统计=============================="""


def show_neuron_layer1_activity():
    T = 8
    H = 128
    labels = ['roll', 'pitch', 'throttle', 'yaw']  # 定义y轴的标签

    # 左上方: 2730, 1010, 2837
    # 右上方: 2564, 113, 3069
    # 右下方: 4574, 4642, 4706
    # 左下方: 361, 2352, 6055, 2402
    # 中心: 1221, 2326, 4820
    images_index = [1010, 2730, 113, 2564, 4706, 4574, 6055, 361, 1221, 2326]
    n_firings = np.max(all_T_spikes1[:, images_index], axis=0).sum(axis=1).astype(int)
    rowspan_list = [2, 5, 1, 2]  # 每一行子图的行数
    grid_size = (sum(rowspan_list), len(images_index))

    plt.figure(figsize=(16, 9))
    font_size = 10
    title_pad = 12

    # 图像
    r = 0
    c = 0
    for i in range(len(images_index)):
        img = all_input[images_index[i]].transpose(1, 2, 0)
        ax_img = plt.subplot2grid(grid_size, (r, c), rowspan=rowspan_list[0], colspan=1)
        ax_img.imshow(img)
        ax_img.axis('off')  # 隐藏坐标轴
        c += 1
    r += rowspan_list[0]

    # 放电栅格图，all_T_spikes1: (T, N, hidden)
    c = 0
    for i in range(len(images_index)):
        ax_spike = plt.subplot2grid(grid_size, (r, c), rowspan=rowspan_list[1], colspan=1)

        # Plotting the raster plot in the specified subplot
        t_indices, neuron_indices = np.where(all_T_spikes1[:, images_index[i]] == 1)
        ax_spike.scatter(t_indices, H - 1 - neuron_indices, marker='.', color='blue')
        ax_spike.set_xlabel('Time Step')
        ax_spike.set_ylabel('Neuron')
        ax_spike.set_xticks(range(T))
        ax_spike.set_yticks(range(H - 1, -1, -H // 16))
        ax_spike.set_yticklabels(range(0, H, H // 16))
        ax_spike.set_ylim(-2, H + 1)
        ax_spike.grid(True, which='both', linestyle='--', linewidth=0.5)
        c += 1
    r += rowspan_list[1]

    # 发放脉冲的神经元数量
    c = 0
    for i in range(len(images_index)):
        plt.subplot2grid(grid_size, (r, c), rowspan=rowspan_list[2], colspan=1)
        plt.text(0.5, 0.5, f"Firings: {n_firings[i]}",
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12, transform=plt.gca().transAxes)
        plt.axis('off')  # Hide the axis for clarity in this demonstration
        c += 1
    r += rowspan_list[2]

    # 输出动作
    c = 0
    for i in range(len(images_index)):
        ax_heatmap = plt.subplot2grid(grid_size, (r, c), rowspan=rowspan_list[3], colspan=1)
        sns_heatmap = sns.heatmap(all_output[images_index[i]].reshape(-1, 1), annot=True, fmt=".3f", cmap='coolwarm',
                                  center=0, vmin=-1, vmax=1, ax=ax_heatmap)
        cbar = sns_heatmap.collections[0].colorbar  # 获取热图的颜色条对象
        cbar.set_ticks([-1, 0, 1])  # 设置颜色条的刻度为[-1, 0, 1]
        ax_heatmap.set_xticks([])  # 隐藏横坐标
        ax_heatmap.set_yticks(ticks=np.arange(len(labels)) + 0.5)
        ax_heatmap.set_yticklabels(labels, rotation=0)
        c += 1
    r += rowspan_list[3]

    plt.tight_layout()
    plt.show()


def draw_firing_counts_by_position():
    # 目标位置和对应的平均发放脉冲神经元数量
    positions = ['Top Left', 'Top Right', 'Bottom Right', 'Bottom Left', 'Center']
    average_firings = [36, 38, 55, 40, 10]
    colors_list = [
        (123, 159, 249),
        (175, 202, 252),
        (246, 189, 162),
        (238, 132, 104),
        (210, 210, 210),
    ]
    colors = [(r / 255, g / 255, b / 255) for r, g, b in colors_list]

    # 设置图表的风格和颜色方案
    plt.style.use('seaborn-v0_8-darkgrid')
    color_palette = plt.get_cmap('coolwarm')

    # 创建柱状图
    plt.figure(figsize=(7, 5))
    bars = plt.bar(positions, average_firings, color=colors)

    # 为每个柱子添加数值标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, yval, ha='center', va='bottom', fontsize=12)

    # 设置图表标题和轴标签
    plt.title('Average Neuron Firings by Target Position in Visual Field', pad=20)
    plt.xlabel('Target Position', labelpad=14)
    plt.ylabel('Average Neuron Firings', labelpad=14)

    # 调整图表布局并显示
    plt.tight_layout()
    plt.show()


def plot_trajectory_with_spike_rate_heatmap_2d(trajectory_coords, neuron_spike_rates, cmap='coolwarm'):
    """
    绘制无人机飞行轨迹的热图，其中轨迹的颜色变化表示了神经元脉冲发放率的变化。

    参数:
    - trajectory_coords: 无人机飞行轨迹坐标的数组，形状为(T, 2)。
    - neuron_spike_rates: 每个时间步的神经元脉冲发放率的数组，长度为T。
    - cmap: 使用的colormap名称。
    """
    T = len(neuron_spike_rates)

    # 创建颜色映射
    color_map = mpl.colormaps.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(T - 1):
        # 计算当前和下一个时间步的坐标
        x = trajectory_coords[i:i + 2, 0]
        y = trajectory_coords[i:i + 2, 1]

        # 获取当前时间步的脉冲发放率，根据脉冲发放率选择颜色
        color = color_map(neuron_spike_rates[i])

        # 绘制这一小段轨迹
        plt.plot(x, y, color=color, linewidth=2)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)  # 调整颜色条的大小和间距
    cbar.set_label('Spike Rate')

    ax.set_title('Drone Flight Trajectory with Neuron Spike Rate Heatmap')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


def plot_trajectory_with_spike_rate_heatmap_3d(ax, trajectory_coords, neuron_spike_rates, cmap='coolwarm'):
    """
    绘制无人机三维飞行轨迹的热图，其中轨迹的颜色变化表示了神经元脉冲发放率的变化。

    参数:
    - trajectory_coords: 无人机飞行轨迹坐标的数组，形状为(T, 3)。
    - neuron_spike_rates: 每个时间步的神经元脉冲发放率的数组，长度为T。
    - cmap: 使用的colormap名称。
    """
    T = len(neuron_spike_rates)

    # 创建颜色映射
    color_map = mpl.colormaps.get_cmap(cmap)

    trajectory_coords = trajectory_coords - trajectory_coords[0, :]  # 将轨迹平移到坐标原点
    for i in range(T - 1):
        # 计算当前和下一个时间步的坐标
        x = trajectory_coords[i:i + 2, 0]
        y = trajectory_coords[i:i + 2, 1]
        z = trajectory_coords[i:i + 2, 2]

        # 获取当前时间步的脉冲发放率，根据脉冲发放率选择颜色
        color = color_map(neuron_spike_rates[i])

        # 绘制这一小段轨迹
        ax.plot(x, y, z, color=color, linewidth=2)

    # 添加颜色条
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])

    # 在起点和终点处绘制特殊标记
    ax.scatter(*trajectory_coords[0, :], color='green', s=90, marker='o', label='Start')
    ax.scatter(*trajectory_coords[-1, :], color='red', s=90, marker='X', label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim((0, 15))
    ax.set_ylim((-10, 10))
    ax.set_zlim((-10, 10))

    # 设置x, y, z轴的刻度间隔
    # ax.set_xticks(np.arange(0, 21, 5))  # 从0到20，间隔为5
    # ax.set_yticks(np.arange(-10, 11, 5))  # 从-10到10，间隔为5
    # ax.set_zticks(np.arange(-10, 11, 5))  # 从-10到10，间隔为5

    ax.invert_yaxis()  # 翻转Y轴
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=10)


def run_trajectory_with_spike_rate_heatmap_3d():
    env_wrapper = DroneEnvWrapper(render=True)

    position_list = []
    all_spike_rate1 = []
    step = 0
    navigation_start_sequence = True
    state, position = env_wrapper.reset()
    position_list.append(position)

    for _ in range(300):  # max time step
        if navigation_start_sequence:
            action = model.predict(state, start_sequence=True)
            navigation_start_sequence = False
        else:
            action = model.predict(state, start_sequence=False)

        all_spike_rate1.extend(spike_seq_monitor1.records)
        spike_seq_monitor1.clear_recorded_data()  # list: [Tensor: (T, batch_size, hidden)]
        v_seq_monitor1.clear_recorded_data()  # list: [Tensor: (T, batch_size, hidden)]
        v_seq_monitor2.clear_recorded_data()  # list: [Tensor: (T, batch_size, hidden)]

        next_state, position, done, successful = env_wrapper.step(action)
        position_list.append(position)

        state = next_state
        step += 1
        if done:
            break

    position_array = np.array(position_list)
    position_array[:, -1] = -position_array[:, -1]  # 反转z轴
    all_spike_rate1 = torch.stack(all_spike_rate1, dim=1).squeeze(2)  # (T, N, hidden), N为所有图像数
    all_spike_rate1 = torch.mean(all_spike_rate1, dim=0).cpu().numpy()  # (N, hidden), N为所有图像数

    def synchronize_view(axs):
        """
        视角同步
        """
        # 获取第一个轴对象的视角
        elev, azim = axs[0].elev, axs[0].azim

        # 同步所有其他轴对象的视角
        for ax in axs[1:]:
            ax.view_init(elev=elev, azim=azim)

        plt.draw()  # 重绘图表

    neuron_index = [2, 3, 74, 1, 102]  # 左，右，上，下，前
    n_neurons = len(neuron_index)
    fig = plt.figure(figsize=(18, 5))

    axs = []  # 存储所有子图的列表
    gs = gridspec.GridSpec(1, n_neurons + 1, width_ratios=[*[1 for _ in range(n_neurons)], 0.5])  # 最后一个数字控制颜色条的宽度

    for i, idx in enumerate(neuron_index):
        ax = fig.add_subplot(gs[i], projection='3d')
        axs.append(ax)  # 将子图添加到列表中
        plot_trajectory_with_spike_rate_heatmap_3d(ax, position_array, all_spike_rate1[:, idx], cmap='coolwarm')
        ax.set_title(f'Neuron {idx}')

    # 在大图旁边添加一个共享的颜色条
    sm = mpl.cm.ScalarMappable(cmap='coolwarm', norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cax = fig.add_axes([0.93, 0.2, 0.01, 0.6])  # [左, 下, 宽, 高]，数值是相对于整个画布的比例
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Spike Rate')

    # 设置定时器，定期同步视角
    timer = fig.canvas.new_timer(interval=500)  # 时间间隔设置为100毫秒
    timer.add_callback(synchronize_view, axs)
    timer.start()

    plt.tight_layout()
    plt.show()


"""==============================SpikingRNN第二层，神经元与输出动作相关系数统计=============================="""


def get_corr_matrix(array1, array2):  # 形状均为(N, H)
    n_h1 = array1.shape[-1]
    n_h2 = array2.shape[-1]
    corr_matrix = np.empty((n_h1, n_h2))  # 初始化相关系数矩阵
    for i in range(n_h1):
        for j in range(n_h2):
            # 检查输入是否是常数数组
            if np.std(array1[:, i]) == 0 or np.std(array2[:, j]) == 0:
                corr_matrix[i, j] = 0
            else:
                corr_matrix[i, j], _ = pearsonr(array1[:, i], array2[:, j])
    return corr_matrix  # (n_h1, n_h2)


def plot_output_neuron_correlation_distributions():
    corr_matrix = get_corr_matrix(all_v_last2, all_output)  # (H, Output)

    # 设置图表的风格和颜色方案
    sns.set(style="darkgrid")
    action_labels = ['Roll', 'Pitch', 'Throttle', 'Yaw']

    # 创建2x2子图
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for i, ax in enumerate(axs.flat):
        # 使用seaborn的kdeplot函数绘制每个输出动作的相关性分布的KDE图
        sns.kdeplot(corr_matrix[:, i], ax=ax, fill=True, color='skyblue', edgecolor='black', linewidth=1.5)
        ax.set_title(action_labels[i], fontsize=14, pad=10)
        ax.set_xlabel('Correlation', fontsize=12, labelpad=10)
        ax.set_ylabel('Probability Density', fontsize=12, labelpad=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)

        ax.set_xlim(-1.5, 1.5)

    # 添加总标题
    fig.suptitle('Neuron Correlation Distributions with Output Actions', fontsize=16)
    plt.tight_layout()
    plt.show()


# show_neuron_layer1_activity()
# draw_firing_counts_by_position()

run_trajectory_with_spike_rate_heatmap_3d()

# plot_output_neuron_correlation_distributions()
