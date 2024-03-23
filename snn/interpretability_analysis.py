import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from sklearn.decomposition import PCA

from spikingjelly import visualizing
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer, monitor

from load_dataset import CustomDataset
from cnn_srnn_model import Model

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

with torch.no_grad():
    for batch_data, _ in tqdm(all_dataloader, file=sys.stdout, leave=False):
        all_input.append(batch_data.squeeze(0))
        batch_data = batch_data.to(model.device)
        batch_output = net(batch_data)  # (batch_size, frames, output)

        all_spike_rate1.extend(spike_seq_monitor1.records)
        all_v_last1.extend(v_seq_monitor1.records)
        all_v_last2.extend(v_seq_monitor2.records)
        all_output.append(batch_output.squeeze(0))

        spike_seq_monitor1.clear_recorded_data()  # list: [Tensor: (T, batch_size, hidden)]
        v_seq_monitor1.clear_recorded_data()  # list: [Tensor: (T, batch_size, hidden)]
        v_seq_monitor2.clear_recorded_data()  # list: [Tensor: (T, batch_size, hidden)]

    all_spike_rate1 = torch.stack(all_spike_rate1, dim=1).squeeze(2)  # (T, N, hidden), N为所有图像数
    all_v_last1 = torch.stack(all_v_last1, dim=1).squeeze(2)  # (T, N, hidden), N为所有图像数
    all_v_last2 = torch.stack(all_v_last2, dim=1).squeeze(2)  # (T, N, hidden), N为所有图像数

    all_T_spikes1 = all_spike_rate1.cpu().numpy()  # (T, N, hidden), N为所有图像数
    all_T_v1 = all_v_last1.cpu().numpy()  # (T, N, hidden), N为所有图像数
    all_T_v2 = all_v_last2.cpu().numpy()  # (T, N, hidden), N为所有图像数
    all_spike_rate1 = torch.mean(all_spike_rate1, dim=0).cpu().numpy()  # (N, hidden), N为所有图像数
    all_v_last2 = all_v_last2[-1, ...].cpu().numpy()  # (N, hidden), N为所有图像数

    all_input = torch.cat(all_input, dim=0).cpu().numpy()  # (N, C, H, W), N为所有图像数
    all_output = torch.cat(all_output, dim=0).cpu().numpy()  # (N, output), N为所有图像数

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
    images_index = [2730, 1010, 2564, 113, 4574, 4706, 361, 6055, 1221, 2326]
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
    plt.figure(figsize=(10, 6))
    bars = plt.bar(positions, average_firings, color=colors)

    # 为每个柱子添加数值标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.4, yval, ha='center', va='bottom', fontsize=14)

    # 设置图表标题和轴标签
    plt.title('Average Neuron Firings by Target Position in Visual Field', pad=20)
    plt.xlabel('Target Position', labelpad=15)
    plt.ylabel('Average Neuron Firings', labelpad=15)

    # 调整图表布局并显示
    plt.tight_layout()
    plt.show()


# show_neuron_layer1_activity()
draw_firing_counts_by_position()
