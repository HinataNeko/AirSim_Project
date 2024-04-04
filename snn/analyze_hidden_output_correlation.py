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


"""==============================相关性矩阵=============================="""


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


def show_corr_matrix():
    corr_matrix1 = get_corr_matrix(all_spike_rate1, all_output)
    corr_matrix2 = get_corr_matrix(all_v_last2, all_output)

    # 可视化
    labels = ['roll', 'pitch', 'throttle', 'yaw']  # 定义y轴的标签

    plt.figure(figsize=(16, 9))
    plt.subplot(2, 1, 1)
    sns.heatmap(corr_matrix1.T, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title("Correlation between Spike Rate of 1st Layer LIF Neurons and Output Actions")
    plt.xlabel("Neuron Index in Hidden Layer")
    plt.ylabel("Output Actions")
    plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0)

    plt.subplot(2, 1, 2)
    sns.heatmap(corr_matrix2.T, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title("Correlation between Last Time Step Membrane Potential of Last Layer LIF Neurons and Output Actions")
    plt.xlabel("Neuron Index in Hidden Layer")
    plt.ylabel("Output Actions")
    plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0)

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图形区域
    plt.show()


"""==============================PCA降维贡献度可视化=============================="""


def show_pca_contribution():
    pca1 = PCA(32)
    spike_rate1_pca = pca1.fit_transform(all_spike_rate1)  # 拟合并转换数据
    spike_rate1_explained_variances = pca1.explained_variance_ratio_  # 查看每个主成分的贡献度
    spike_rate1_cumulative_variances = np.cumsum(spike_rate1_explained_variances)  # 计算累计贡献度

    pca2 = PCA(32)
    v_last2_pca = pca2.fit_transform(all_v_last2)  # 拟合并转换数据
    v_last2_explained_variances = pca2.explained_variance_ratio_  # 查看每个主成分的贡献度
    v_last2_cumulative_variances = np.cumsum(v_last2_explained_variances)  # 计算累计贡献度

    # 可视化每个主成分的贡献度
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    # 绘制每个主成分的贡献度柱状图
    plt.bar(range(1, len(spike_rate1_explained_variances) + 1), spike_rate1_explained_variances,
            align='center', label='Individual explained variance')

    # 绘制累计贡献度的折线图
    plt.plot(range(1, len(spike_rate1_cumulative_variances) + 1), spike_rate1_cumulative_variances,
             marker='o', linestyle='-', color='red', markersize=4, label='Cumulative explained variance')

    plt.title('PCA Explained Variance: Spike Rate of 1st Layer LIF Neurons')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.ylim([0, 1])
    # plt.legend(loc='best')
    plt.grid()

    plt.subplot(1, 2, 2)

    # 绘制每个主成分的贡献度柱状图
    plt.bar(range(1, len(v_last2_explained_variances) + 1), v_last2_explained_variances,
            align='center', label='Individual explained variance')

    # 绘制累计贡献度的折线图
    plt.plot(range(1, len(v_last2_cumulative_variances) + 1), v_last2_cumulative_variances,
             marker='o', linestyle='-', color='red', markersize=4, label='Cumulative explained variance')

    plt.title('PCA Explained Variance: Last Time Step Membrane Potential of Last Layer LIF Neurons')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.ylim([0, 1])
    plt.legend(loc='best')
    plt.grid()

    plt.tight_layout()
    plt.show()


"""==============================PCA降维后相关性矩阵=============================="""


def show_corr_matrix_after_pca():
    spike_rate1_pca_model = PCA(2)
    spike_rate1_pca_2d = spike_rate1_pca_model.fit_transform(all_spike_rate1)  # (N, H)->(N, 2)
    spike_rate1_pca_corr_matrix = get_corr_matrix(spike_rate1_pca_2d, all_output)

    v_last2_pca_model = PCA(2)
    v_last2_pca_2d = v_last2_pca_model.fit_transform(all_v_last2)  # (N, H)->(N, 2)
    v_last2_pca_corr_matrix = get_corr_matrix(v_last2_pca_2d, all_output)

    # 可视化
    labels = ['roll', 'pitch', 'throttle', 'yaw']  # 定义y轴的标签
    pc_labels = ['PC1', 'PC2']  # 定义x轴的标签

    plt.figure(figsize=(16, 9))
    plt.subplot(2, 1, 1)
    sns.heatmap(spike_rate1_pca_corr_matrix.T, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title("Correlation between Spike Rate of 1st Layer after PCA")
    plt.xlabel("PCA Components")
    plt.ylabel("Output Actions")
    plt.xticks(ticks=np.arange(len(pc_labels)) + 0.5, labels=pc_labels, rotation=0)
    plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0)

    plt.subplot(2, 1, 2)
    sns.heatmap(v_last2_pca_corr_matrix.T, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title("Correlation between Last Time Step Membrane Potential after PCA")
    plt.xlabel("PCA Components")
    plt.ylabel("Output Actions")
    plt.xticks(ticks=np.arange(len(pc_labels)) + 0.5, labels=pc_labels, rotation=0)
    plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0)

    plt.tight_layout()
    plt.show()


"""==============================部分神经元活动=============================="""


# all_input: (N, C, H, W)
# all_output: (N, output)
# all_T_spikes1: (T, N, hidden)
# all_T_v1: (T, N, hidden)
# all_T_v2: (T, N, hidden)
# all_v_last2: (N, hidden)

#  sr1[3]--yaw--正相关
#  sr1[2]--yaw--负相关
#  sr1[74]--throttle--正相关
#  sr1[1]--throttle--负相关

#  v2[0]--yaw--正相关
#  v2[2]--yaw--负相关
#  v2[15]--throttle--正相关
#  v2[74]--throttle--负相关

# 播放视频
def play_dataset_as_video(start_frame=0):
    cv2.destroyAllWindows()
    idx = start_frame
    while 0 <= idx < len(all_input):
        frame = np.transpose(all_input[idx], (1, 2, 0))

        # 上采样（放大2倍），使用线性插值
        height, width = frame.shape[:2]
        frame_resized = cv2.resize(frame, (2 * width, 2 * height))
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)

        # 将帧号作为文本添加到图像
        cv2.putText(frame_resized, f'Frame: {idx}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Video', frame_resized)

        key = cv2.waitKey(0)

        if key == 32:  # 空格键播放
            idx += 1
        elif key == 110:  # N键跳10帧
            idx += 10
        elif key in [98, 66]:  # "B" 或 "b" 键回退
            idx = max(0, idx - 1)  # 防止索引小于0

    cv2.destroyAllWindows()


def show_neuron_layer2_activity():
    T = 8
    t = np.arange(0, T)
    v_threshold = None
    labels = ['roll', 'pitch', 'throttle', 'yaw']  # 定义y轴的标签

    # images_index = [0, 112, 1127, 2204, 2352, 3492, 4574]
    images_index = [3492, 4574, 2352, 180]
    grid_size = (len(images_index), 11)
    colors = [['red', 'blue', 'red', 'blue'],
              ['red', 'blue', 'blue', 'red'],
              ['blue', 'red', 'blue', 'red'],
              ['blue', 'red', 'red', 'blue'],
              ['red', 'blue', 'red', 'blue']]
    plt.figure(figsize=(16, 9))
    font_size = 10
    title_pad = 12

    for i in range(len(images_index)):
        c = 0
        img = all_input[images_index[i]].transpose(1, 2, 0)
        ax_img = plt.subplot2grid(grid_size, (i, c), colspan=2)
        ax_img.imshow(img)
        ax_img.axis('off')  # 隐藏坐标轴
        if i == 0:
            ax_img.set_title(f'Input Image', fontsize=font_size, pad=title_pad)  # 为图片设置标题
        c += 2

        ax_heatmap = plt.subplot2grid(grid_size, (i, c))
        sns_heatmap = sns.heatmap(all_output[images_index[i]].reshape(-1, 1), annot=True, fmt=".3f", cmap='coolwarm',
                                  center=0, vmin=-1, vmax=1, ax=ax_heatmap)
        cbar = sns_heatmap.collections[0].colorbar  # 获取热图的颜色条对象
        cbar.set_ticks([-1, 0, 1])  # 设置颜色条的刻度为[-1, 0, 1]
        ax_heatmap.set_xticks([])  # 隐藏横坐标
        ax_heatmap.set_yticks(ticks=np.arange(len(labels)) + 0.5)
        ax_heatmap.set_yticklabels(labels, rotation=0)
        if i == 0:
            ax_heatmap.set_title('Output Heatmap', fontsize=font_size, pad=title_pad)
        c += 1

        ax0 = plt.subplot2grid(grid_size, (i, c), colspan=2)
        ax0.plot(t, all_T_v2[:, images_index[i], 0], color=colors[i][0])
        ax0.set_xlim(-0.5, T - 0.5)
        ax0.set_xticks(range(T))  # 设置横轴刻度位置
        # ax0.set_xticklabels(range(T))  # 设置横轴刻度标签
        ax0.text(1.04, -0.12, 't', va='center', ha='center', transform=ax0.transAxes,
                 fontsize=font_size)  # 设置横轴标题位置为横轴的右侧
        ax0.text(-0.05, 1.05, 'v', va='center', ha='center', transform=ax0.transAxes, fontsize=font_size)
        if v_threshold:
            ax0.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')
        if i == 0:
            ax0.set_title(r"Neuron 0 ($\bf{Yaw\ Positive}$ Correlation)", fontsize=font_size, pad=title_pad)
        c += 2

        ax0 = plt.subplot2grid(grid_size, (i, c), colspan=2)
        ax0.plot(t, all_T_v2[:, images_index[i], 2], color=colors[i][1])
        ax0.set_xlim(-0.5, T - 0.5)
        ax0.set_xticks(range(T))  # 设置横轴刻度位置
        ax0.text(1.04, -0.12, 't', va='center', ha='center', transform=ax0.transAxes,
                 fontsize=font_size)  # 设置横轴标题位置为横轴的右侧
        ax0.text(-0.05, 1.05, 'v', va='center', ha='center', transform=ax0.transAxes, fontsize=font_size)
        if v_threshold:
            ax0.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')
        if i == 0:
            ax0.set_title(r"Neuron 2 ($\bf{Yaw\ Negative}$ Correlation)", fontsize=font_size, pad=title_pad)
        c += 2

        ax0 = plt.subplot2grid(grid_size, (i, c), colspan=2)
        ax0.plot(t, all_T_v2[:, images_index[i], 15], color=colors[i][2])
        ax0.set_xlim(-0.5, T - 0.5)
        ax0.set_xticks(range(T))  # 设置横轴刻度位置
        ax0.text(1.04, -0.12, 't', va='center', ha='center', transform=ax0.transAxes,
                 fontsize=font_size)  # 设置横轴标题位置为横轴的右侧
        ax0.text(-0.05, 1.05, 'v', va='center', ha='center', transform=ax0.transAxes, fontsize=font_size)
        if v_threshold:
            ax0.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')
        if i == 0:
            ax0.set_title(r"Neuron 15 ($\bf{Throttle\ Positive}$ Correlation)", fontsize=font_size, pad=title_pad)
        c += 2

        ax0 = plt.subplot2grid(grid_size, (i, c), colspan=2)
        ax0.plot(t, all_T_v2[:, images_index[i], 74], color=colors[i][3])
        ax0.set_xlim(-0.5, T - 0.5)
        ax0.set_xticks(range(T))  # 设置横轴刻度位置
        ax0.text(1.04, -0.12, 't', va='center', ha='center', transform=ax0.transAxes,
                 fontsize=font_size)  # 设置横轴标题位置为横轴的右侧
        ax0.text(-0.05, 1.05, 'v', va='center', ha='center', transform=ax0.transAxes, fontsize=font_size)
        if v_threshold:
            ax0.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')
        if i == 0:
            ax0.set_title(r"Neuron 74 ($\bf{Throttle\ Negative}$ Correlation)", fontsize=font_size, pad=title_pad)
        c += 2

    plt.tight_layout()
    plt.show()


def show_neuron_layer1_activity():
    T = 8
    H = 128
    labels = ['roll', 'pitch', 'throttle', 'yaw']  # 定义y轴的标签

    # start_frame = 3680
    # interval = 5
    # images_index = [start_frame + i * interval for i in range(7)]
    images_index = [3680, 3686, 3694, 3710, 3718, 3725]
    rowspan_list = [3, 5, 5, 2]  # 每一行子图的行数
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
        if i == 0:
            ax_img.text(-0.5, 0.5, r'$\bf{Input\ Images}$', transform=ax_img.transAxes,
                        ha='right', va='center', fontsize=font_size)
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
        if i == 0:
            ax_spike.text(-0.4, 0.5, r'$\bf{Spikes\ Raster}$', transform=ax_spike.transAxes,
                          ha='right', va='center', fontsize=font_size)
    r += rowspan_list[1]

    # 脉冲发放率
    c = 0
    for i in range(len(images_index)):
        ax_heatmap = plt.subplot2grid(grid_size, (r, c), rowspan=rowspan_list[2], colspan=1)
        firing_rates = np.mean(all_T_spikes1[:, images_index[i]], axis=0)
        sns_heatmap = sns.heatmap(firing_rates.reshape(-1, 1), annot=False, fmt=".3f", cmap='coolwarm',
                                  center=0.5, vmin=0, vmax=1, ax=ax_heatmap)
        cbar = sns_heatmap.collections[0].colorbar  # 获取热图的颜色条对象
        cbar.set_ticks([0, 0.5, 1])  # 设置颜色条的刻度
        ax_heatmap.set_xticks([])  # 隐藏横坐标
        ax_heatmap.set_yticks(range(0, H, H // 16))
        ax_heatmap.set_yticklabels(range(0, H, H // 16))
        ax_heatmap.set_ylabel('Neuron')
        c += 1
        if i == 0:
            ax_heatmap.text(-0.5, 0.5, r'$\bf{Firing\ Rates}$', transform=ax_heatmap.transAxes,
                            ha='right', va='center', fontsize=font_size)
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
        if i == 0:
            ax_heatmap.text(-0.5, 0.5, r'$\bf{Outputs}$', transform=ax_heatmap.transAxes,
                            ha='right', va='center', fontsize=font_size)
    r += rowspan_list[3]

    plt.tight_layout()
    plt.show()


def show_relation_scatter():
    labels = ['roll', 'pitch', 'throttle', 'yaw']
    neuron_index = [0, 114, 15, 0]  # 指定的神经元索引

    # 绘图
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))  # 创建2行2列的子图

    for i in range(4):  # 遍历每个输出动作
        row = i // 2  # 计算当前子图的行索引
        col = i % 2   # 计算当前子图的列索引
        axs[row, col].scatter(all_output[:, i], all_v_last2[:, neuron_index[i]], alpha=0.5)  # 绘制散点图
        axs[row, col].set_title(f'{labels[i]} - Neuron {neuron_index[i]}')
        axs[row, col].set_xlabel('Output Value')
        axs[row, col].set_ylabel('Neuron State Value')
        axs[row, col].set_xlim([-1, 1])  # 设置横轴范围

    plt.tight_layout()
    plt.show()


# play_dataset_as_video(0)
# show_neuron_layer1_activity()
show_neuron_layer2_activity()
# show_relation_scatter()
exit()

images_index = [3492, 4574, 2352, 180, 1127]
grid_size = (len(images_index), 11)
colors = [['red', 'blue', 'red', 'blue'],
          ['red', 'blue', 'blue', 'red'],
          ['blue', 'red', 'blue', 'red'],
          ['blue', 'red', 'red', 'blue'],
          ['red', 'blue', 'red', 'blue']]
plt.figure(figsize=(16, 9))
font_size = 10
title_pad = 12

for i in range(len(images_index)):
    c = 0
    img = all_input[images_index[i]].transpose(1, 2, 0)
    ax_img = plt.subplot2grid(grid_size, (i, c), colspan=2)
    ax_img.imshow(img)
    ax_img.axis('off')  # 隐藏坐标轴
    if i == 0:
        ax_img.set_title(f'Input Image', fontsize=font_size, pad=title_pad)  # 为图片设置标题
    c += 2

    ax_heatmap = plt.subplot2grid(grid_size, (i, c))
    sns_heatmap = sns.heatmap(all_output[images_index[i]].reshape(-1, 1), annot=True, fmt=".3f", cmap='coolwarm',
                              center=0, vmin=-1, vmax=1, ax=ax_heatmap)
    cbar = sns_heatmap.collections[0].colorbar  # 获取热图的颜色条对象
    cbar.set_ticks([-1, 0, 1])  # 设置颜色条的刻度为[-1, 0, 1]
    ax_heatmap.set_xticks([])  # 隐藏横坐标
    ax_heatmap.set_yticks(ticks=np.arange(len(labels)) + 0.5)
    ax_heatmap.set_yticklabels(labels, rotation=0)
    if i == 0:
        ax_heatmap.set_title('Output Heatmap', fontsize=font_size, pad=title_pad)
    c += 1

    ax0 = plt.subplot2grid(grid_size, (i, c), colspan=2)
    ax0.plot(t, all_T_v1[:, images_index[i], 3], color=colors[i][0])
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.set_xticks(range(T))  # 设置横轴刻度位置
    # ax0.set_xticklabels(range(T))  # 设置横轴刻度标签
    ax0.text(1.04, -0.12, 't', va='center', ha='center', transform=ax0.transAxes, fontsize=font_size)  # 设置横轴标题位置为横轴的右侧
    ax0.text(-0.05, 1.05, 'v', va='center', ha='center', transform=ax0.transAxes, fontsize=font_size)
    if v_threshold:
        ax0.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')
    if i == 0:
        ax0.set_title(r"Neuron 3 ($\bf{Yaw\ Positive}$ Correlation)", fontsize=font_size, pad=title_pad)
    c += 2

    ax0 = plt.subplot2grid(grid_size, (i, c), colspan=2)
    ax0.plot(t, all_T_v1[:, images_index[i], 2], color=colors[i][1])
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.set_xticks(range(T))  # 设置横轴刻度位置
    ax0.text(1.04, -0.12, 't', va='center', ha='center', transform=ax0.transAxes, fontsize=font_size)  # 设置横轴标题位置为横轴的右侧
    ax0.text(-0.05, 1.05, 'v', va='center', ha='center', transform=ax0.transAxes, fontsize=font_size)
    if v_threshold:
        ax0.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')
    if i == 0:
        ax0.set_title(r"Neuron 2 ($\bf{Yaw\ Negative}$ Correlation)", fontsize=font_size, pad=title_pad)
    c += 2

    ax0 = plt.subplot2grid(grid_size, (i, c), colspan=2)
    ax0.plot(t, all_T_v1[:, images_index[i], 74], color=colors[i][2])
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.set_xticks(range(T))  # 设置横轴刻度位置
    ax0.text(1.04, -0.12, 't', va='center', ha='center', transform=ax0.transAxes, fontsize=font_size)  # 设置横轴标题位置为横轴的右侧
    ax0.text(-0.05, 1.05, 'v', va='center', ha='center', transform=ax0.transAxes, fontsize=font_size)
    if v_threshold:
        ax0.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')
    if i == 0:
        ax0.set_title(r"Neuron 74 ($\bf{Throttle\ Positive}$ Correlation)", fontsize=font_size, pad=title_pad)
    c += 2

    ax0 = plt.subplot2grid(grid_size, (i, c), colspan=2)
    ax0.plot(t, all_T_v1[:, images_index[i], 1], color=colors[i][3])
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.set_xticks(range(T))  # 设置横轴刻度位置
    ax0.text(1.04, -0.12, 't', va='center', ha='center', transform=ax0.transAxes, fontsize=font_size)  # 设置横轴标题位置为横轴的右侧
    ax0.text(-0.05, 1.05, 'v', va='center', ha='center', transform=ax0.transAxes, fontsize=font_size)
    if v_threshold:
        ax0.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')
    if i == 0:
        ax0.set_title(r"Neuron 1 ($\bf{Throttle\ Negative}$ Correlation)", fontsize=font_size, pad=title_pad)
    c += 2

plt.tight_layout()
plt.show()
