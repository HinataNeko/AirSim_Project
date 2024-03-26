import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
import sys
from tqdm import tqdm

from load_dataset import CustomDataset

from evaluation.cnn_rnn_model import Model as Model_RNN
from evaluation.cnn_lstm_model import Model as Model_LSTM
from evaluation.cnn_gru_model import Model as Model_GRU
from snn.cnn_srnn_model import Model as Model_SRNN


def visual_backprop(model, x, add_hook=True):
    feature_maps = []
    model.eval()
    model.to('cpu')

    def hook_fn(module, input, output):
        # 保存ReLU激活后的特征图
        feature_maps.append(output)

    # 注册hook
    if add_hook:
        for layer in model.cnn:
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(hook_fn)

    # 前向传播，获取特征图
    model(x)

    # 反转特征图列表，以便从最深层开始处理
    feature_maps.reverse()

    # 初始化遮罩为最后一个特征图的平均值
    mask = torch.mean(feature_maps[0], dim=1, keepdim=True)
    for i in range(1, len(feature_maps)):
        # 上采样遮罩
        mask = F.interpolate(mask, size=feature_maps[i].size()[2:], mode='nearest')

        # 乘以当前层的平均特征图
        mask *= torch.mean(feature_maps[i], dim=1, keepdim=True)

    # 上采样遮罩到原始输入图像大小
    mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear', align_corners=False)

    # 归一化遮罩
    mask = (mask - mask.min()) / (mask.max() - mask.min())

    # 清空特征图列表
    feature_maps.clear()

    return mask


def show_image_with_mask(model, input_img, add_hook=True, show=True):
    mask = visual_backprop(model.model.cnn, input_img.to(model.device), add_hook=add_hook)
    # 现在，mask包含了输入图像每个像素的贡献度

    mask_np = mask.squeeze().cpu().detach().numpy()  # 将形状转换为 (240, 320)
    image_np = input_img.squeeze().cpu().detach().numpy().transpose(1, 2, 0)  # 将形状转换为 (240, 320, 3)

    mask_np = (mask_np * 255).astype(np.uint8)
    image_np = (image_np * 255).astype(np.uint8)

    # 将遮罩转换为彩色图像以便可视化
    mask_color = cv2.applyColorMap(mask_np, cv2.COLORMAP_JET)

    if show:
        # 垂直堆叠原始图像和遮罩
        combined_image = cv2.vconcat([cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), mask_color])
        cv2.imshow(f'VisualBackProp-{model.name}', combined_image)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), mask_color  # BGR format


if __name__ == '__main__':
    model_rnn = Model_RNN(load_dataset=False)
    model_rnn.load()
    model_rnn.device = 'cpu'
    model_lstm = Model_LSTM(load_dataset=False)
    model_lstm.load()
    model_lstm.device = 'cpu'
    model_gru = Model_GRU(load_dataset=False)
    model_gru.load()
    model_gru.device = 'cpu'
    model_srnn = Model_SRNN(load_dataset=False)
    model_srnn.load()
    model_srnn.device = 'cpu'

    all_dataset = CustomDataset("./datasets/uav_recording", enhance=False, random_clip=False)
    all_dataloader = DataLoader(all_dataset, batch_size=1, shuffle=False)

    all_input = []
    for batch_data, _ in tqdm(all_dataloader, file=sys.stdout, leave=False):
        all_input.append(batch_data.squeeze(0))
    all_input = torch.cat(all_input, dim=0).cpu()  # (N, C, H, W), N为所有图像数

    # [2, 25], [4, 25]
    # [14, 20]
    # [39, 1]
    # [64, 10]
    # for _ in range(76):
    #     input_img, _ = next(all_dataloader_iter)
    # input_img = input_img[:, 10]  # 想要可视化的输入图像
    for i in tqdm(range(0, all_input.shape[0]), file=sys.stdout, leave=False):
        # add_hook = True if i == 0 else False
        add_hook = True
        input_img = all_input[i].unsqueeze(0)  # input_img: (1, 3, 240, 320), torch.float32
        bgr_img, rnn_mask = show_image_with_mask(model_rnn, input_img, add_hook=add_hook, show=False)
        _, lstm_mask = show_image_with_mask(model_lstm, input_img, add_hook=add_hook, show=False)
        _, gru_mask = show_image_with_mask(model_gru, input_img, add_hook=add_hook, show=False)
        _, srnn_mask = show_image_with_mask(model_srnn, input_img, add_hook=add_hook, show=False)

        # 垂直堆叠原始图像和遮罩
        separator_image = np.full((5, bgr_img.shape[1], 3), (255, 255, 255), dtype=np.uint8)
        combined_image = cv2.vconcat([bgr_img, separator_image, rnn_mask, separator_image, lstm_mask, separator_image,
                                      gru_mask, separator_image, srnn_mask])
        # cv2.imshow('VisualBackProp', combined_image)
        cv2.imwrite(f'./images/scene{i}.png', combined_image)
        # cv2.waitKey(0)
