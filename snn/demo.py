import torch
from spikingjelly.activation_based import neuron
from spikingjelly import visualizing
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd

# 步骤2：读取CSV文件
df = pd.read_csv('./Default Dataset.csv')  # 请将'路径/文件名.csv'替换为您的文件路径和名称

# 步骤3：提取x和y坐标
x = df.iloc[:, 0]  # 假设x坐标在第一列
y = df.iloc[:, 1]  # 假设y坐标在第二列

plt.plot(x, y, linewidth=4)
plt.axhline(y=1, color='r', linestyle='--')
plt.show()
exit()

if_layer = neuron.LIFNode(tau=2., v_reset=None)

if_layer.reset()
x = torch.ones(50) * 0.9
T = x.shape[0]
s_list = []
v_list = []
for t in range(T):
    s_list.append(if_layer(x[t].reshape(-1)))
    v_list.append(if_layer.v)

dpi = 100
figsize = (6, 6)
visualizing.plot_one_neuron_v_s(torch.cat(v_list).numpy(), torch.cat(s_list).numpy(), v_threshold=if_layer.v_threshold,
                                v_reset=if_layer.v_reset,
                                figsize=figsize, dpi=dpi)
plt.show()
exit()

if_layer.reset()
T = 100
x = torch.rand([32]) / 0.9
s_list = []
v_list = []
for t in range(T):
    s_list.append(if_layer(x).unsqueeze(0))
    v_list.append(if_layer.v.unsqueeze(0))

s_list = torch.cat(s_list)
v_list = torch.cat(v_list)

figsize = (12, 8)
dpi = 100
visualizing.plot_2d_heatmap(array=v_list.numpy(), title='membrane potentials', xlabel='simulating step',
                            ylabel='neuron index', int_x_ticks=True, x_max=T, figsize=figsize, dpi=dpi)

visualizing.plot_1d_spikes(spikes=s_list.numpy(), title='membrane sotentials', xlabel='simulating step',
                           ylabel='neuron index', figsize=figsize, dpi=dpi)

plt.show()
