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
from sklearn.preprocessing import StandardScaler

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
all_v_last2 = []
all_output = []

# 神经元构造参数中的 store_v_seq: bool = False 表示在默认情况下，只记录当前时刻的电压，不记录所有时刻的电压序列
# 现在我们想记录所有时刻的电压，则将其更改为 True
for m in net.modules():
    if isinstance(m, neuron.LIFNode):
        m.store_v_seq = True

# SpikingRNN第一层
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
        all_v_last2.extend(v_seq_monitor2.records)
        all_output.append(batch_output.squeeze(0))

        spike_seq_monitor1.clear_recorded_data()  # list: [Tensor: (T, batch_size, hidden)]
        v_seq_monitor2.clear_recorded_data()  # list: [Tensor: (T, batch_size, hidden)]

    all_spike_rate1 = torch.stack(all_spike_rate1, dim=1).squeeze(2)  # (T, N, hidden), N为所有图像数
    all_v_last2 = torch.stack(all_v_last2, dim=1).squeeze(2)  # (T, N, hidden), N为所有图像数

    all_T_spikes1 = all_spike_rate1.cpu().numpy()  # (T, N, hidden), N为所有图像数
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
# all_T_v2: (T, N, hidden)

#  sr1[3]--yaw--正相关
#  sr1[2]--yaw--负相关
#  sr1[74]--throttle--正相关
#  sr1[1]--throttle--负相关

#  v2[0]--yaw--正相关
#  v2[2]--yaw--负相关
#  v2[15]--throttle--正相关
#  v2[74]--throttle--负相关

# 播放视频
cv2.destroyAllWindows()
idx = 0
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
    elif key in [98, 66]:  # "B" 或 "b" 键回退
        idx = max(0, idx - 1)  # 防止索引小于0

cv2.destroyAllWindows()

T = 8
t = np.arange(0, T)
v_threshold = 1.

images_index = [0, 112, 300, 1000, 1100]
grid_size = (len(images_index), 9)

plt.figure(figsize=(16, 9))

for i in range(len(images_index)):
    img = all_input[images_index[i]].transpose(1, 2, 0)
    ax0 = plt.subplot2grid(grid_size, (i, 0))
    ax0.imshow(img)
    ax0.axis('off')  # 隐藏坐标轴

    ax0 = plt.subplot2grid(grid_size, (i, 1), colspan=2)
    ax0.plot(t, all_T_v2[:, images_index[i], 0])
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')

    ax0 = plt.subplot2grid(grid_size, (i, 3), colspan=2)
    ax0.plot(t, all_T_v2[:, images_index[i], 2])
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')

    ax0 = plt.subplot2grid(grid_size, (i, 5), colspan=2)
    ax0.plot(t, all_T_v2[:, images_index[i], 15])
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')

    ax0 = plt.subplot2grid(grid_size, (i, 7), colspan=2)
    ax0.plot(t, all_T_v2[:, images_index[i], 74])
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')

plt.tight_layout()
plt.show()
