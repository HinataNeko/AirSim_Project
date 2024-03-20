import torch
from spikingjelly.activation_based import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt

if_layer = neuron.LIFNode(tau=2.)
if_layer.reset()

T = 100
# input_x = torch.ones([T, 1]) * (-0.5)
input_x = torch.zeros([T, 1])
input_x[10:20] = 1.1

s_list = []
v_list = []
for t in range(T):
    s_list.append(if_layer(input_x[t]))
    v_list.append(if_layer.v)

# dpi = 150
# figsize = (12, 8)
# visualizing.plot_one_neuron_v_s(torch.cat(v_list).numpy(), torch.cat(s_list).numpy(), v_threshold=if_layer.v_threshold,
#                                 v_reset=if_layer.v_reset,
#                                 figsize=figsize, dpi=dpi)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(input_x.squeeze(1).numpy())
plt.title("Input")

plt.subplot(2, 1, 2)
plt.plot(torch.cat(v_list).numpy())
plt.title("V")

plt.show()
