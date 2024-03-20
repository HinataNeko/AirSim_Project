import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2

from cnn_ncps_model import Model as Model_CfC
from cnn_lstm_model import Model as Model_LSTM
from cnn_srnn_model import Model as Model_SRNN


def visual_backprop(model, x):
    feature_maps = []
    model.eval()

    def hook_fn(module, input, output):
        # 保存ReLU激活后的特征图
        feature_maps.append(output)

    # 注册hook
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

    return mask


def show_image_with_mask(model, input_img):
    mask = visual_backprop(model.model.cnn, input_img.to(model.device))
    # 现在，mask包含了输入图像每个像素的贡献度

    mask_np = mask.squeeze().cpu().detach().numpy()  # 将形状转换为 (240, 320)
    image_np = input_img.squeeze().cpu().detach().numpy().transpose(1, 2, 0)  # 将形状转换为 (240, 320, 3)

    mask_np = (mask_np * 255).astype(np.uint8)
    image_np = (image_np * 255).astype(np.uint8)

    # 将遮罩转换为彩色图像以便可视化
    mask_color = cv2.applyColorMap(mask_np, cv2.COLORMAP_JET)

    # 垂直堆叠原始图像和遮罩
    combined_image = cv2.vconcat([cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), mask_color])

    # 显示最终结果或保存到文件
    cv2.imshow(f'VisualBackProp-{model.name}', combined_image)


if __name__ == '__main__':
    model_cfc = Model_CfC(load_dataset=True)
    model_cfc.load()
    model_lstm = Model_LSTM(load_dataset=True)
    model_lstm.load()
    model_srnn = Model_SRNN(load_dataset=True)
    model_srnn.load()

    dataloader_iter = iter(model_cfc.test_dataloader)

    # [2, 25]
    # [4, 25]
    # [14, 17]
    # [39, 1]
    # [64, 10]
    for _ in range(35):
        input_img, _ = next(dataloader_iter)
    input_img = input_img[:, 1]  # 想要可视化的输入图像

    show_image_with_mask(model_cfc, input_img)  # input_img: (1, 3, 240, 320)
    show_image_with_mask(model_lstm, input_img)  # input_img: (1, 3, 240, 320)
    show_image_with_mask(model_srnn, input_img)  # input_img: (1, 3, 240, 320)

    cv2.waitKey(0)
