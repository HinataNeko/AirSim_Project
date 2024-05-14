import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import normalize, resize, to_pil_image
from ncps.torch import CfC

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm

from load_dataset import CustomDataset


class PrintLayer(nn.Module):
    def __init__(self, label=""):
        super(PrintLayer, self).__init__()
        self.label = label

    def forward(self, x):
        print(f"{self.label} Shape: {x.shape}")
        return x


class BaseCNN(nn.Module):
    def __init__(self, hidden_dim=128, out_features=128):
        super(BaseCNN, self).__init__()
        self.cnn = nn.Sequential(  # 240*320
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),  # 118*158
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),  # 57*77
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),  # 27*37
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),  # 12*17
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=hidden_dim, kernel_size=5, stride=2),
            nn.BatchNorm2d(hidden_dim),  # 4*7
            nn.ReLU(),
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(  # 128*4*7
            nn.Linear(in_features=3584, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=out_features),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


class CNNtoRNN(nn.Module):
    def __init__(self, cnn_hidden=128, cnn_output_features=128, rnn_hidden_size=128, output_size=4):
        super(CNNtoRNN, self).__init__()
        self.cnn_output_features = cnn_output_features
        self.rnn_hidden_size = rnn_hidden_size
        self.output_size = output_size

        self.cnn = BaseCNN(cnn_hidden, cnn_output_features)
        self.rnn = CfC(self.cnn_output_features, self.rnn_hidden_size, batch_first=True)
        self.output_layer = nn.Sequential(
            nn.Linear(self.rnn_hidden_size, self.output_size),
            nn.Tanh(),
        )
        self.hidden = None  # RNN之前的状态

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _reset_hidden(self):
        self.hidden = torch.zeros(1, self.rnn_hidden_size).to(self.device)

    def forward(self, img_frames):  # (batch_size, frames, channels, height, width)
        batch_size, frames, channels, height, width = img_frames.shape
        img_frames = img_frames.view(-1, channels, height, width)
        cnn_output = self.cnn(img_frames)
        cnn_output = cnn_output.view(batch_size, frames, self.cnn_output_features)
        current_batch_size = cnn_output.shape[0]

        h_0 = torch.zeros(current_batch_size, self.rnn_hidden_size).to(self.device)  # (batch, units)

        # rnn_output: (batch_size, frames, num_directions * hidden_size)
        rnn_output, h_n = self.rnn(cnn_output, h_0)
        output = self.output_layer(rnn_output)

        return output

    def predict(self, input_img, start_sequence=True):  # (channels, height, width)
        if start_sequence:
            # Reset hidden and cell states
            self._reset_hidden()

        input_img = input_img.unsqueeze(0)  # (1, 1, channels, height, width)
        cnn_output = self.cnn(input_img)
        cnn_output = cnn_output.view(1, 1, self.cnn_output_features)  # Reshape for LSTM

        # Process the CNN output through the LSTM
        rnn_output, self.hidden = self.rnn(cnn_output, self.hidden)
        output = self.output_layer(rnn_output)

        prediction = output.squeeze(0).squeeze(0)
        return prediction


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input, target):
        if input.size() != target.size():
            raise ValueError("Input and target must have the same shape")

        # 将权重调整到正确的形状并扩展以匹配输入
        # roll, pitch, thrust, yaw
        weights = torch.ones_like(input)
        weights[:, :, 2:] = 2.
        weights[:, :15, :] *= 2.

        # 计算加权的平方差
        loss = weights * (input - target) ** 2

        # 返回损失的均值或总和
        return loss.mean()  # 或者使用 loss.sum()


class Model:
    def __init__(self, load_dataset=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = "CNN_CfC_model"
        self.model = CNNtoRNN().to(self.device)
        self.cost = WeightedMSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.checkpoint = {'model_state_dict': self.model.state_dict(),
                           'optimizer_state_dict': self.optimizer.state_dict(), }
        self.model_save_path = f'./saved_model/{self.name}.pth'

        self.input_size = (240, 320)
        self.input_shape = (3, *self.input_size)

        if load_dataset:
            self.train_dataset = CustomDataset("./datasets/uav_recording", enhance=True, random_clip=True)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=4, shuffle=True)
            self.test_dataset = CustomDataset("./datasets/uav_recording", enhance=False, random_clip=False)
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def save(self):
        torch.save(self.checkpoint, self.model_save_path)
        print("Model saved to：" + self.model_save_path)

    def load(self, path=None):
        if path is None:
            path = self.model_save_path
        if os.path.isfile(path):  # 存在之前以保存的模型
            model_data = torch.load(path)
            self.model.load_state_dict(model_data['model_state_dict'])  # 模型数据
            self.optimizer.load_state_dict(model_data['optimizer_state_dict'])  # 优化器数据
            print("Previous model loaded：" + path)

    def predict(self, rgb_img, transform=True, start_sequence=True):
        if transform:
            input_img = torch.tensor(rgb_img.astype(np.float32) / 255.).permute((2, 0, 1))  # (channels, height, width)
        else:
            input_img = rgb_img

        self.model.eval()
        with torch.no_grad():
            output = self.model.predict(input_img.to(self.device), start_sequence)

        return output.cpu().numpy()

    def train(self, epochs, startEpoch=1, val=True):
        if epochs <= 0:
            return
        print(f"开始训练{epochs}次")
        train_loss_list = []

        for epoch in range(startEpoch, startEpoch + epochs):
            train_loss = 0
            self.model.train()  # 训练模式

            for batch_data, batch_labels in tqdm(self.train_dataloader, file=sys.stdout, leave=False):
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                batch_output = self.model(batch_data)
                batch_loss = self.cost(batch_output, batch_labels)
                train_loss += batch_loss.item()

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            train_loss /= len(self.train_dataloader)  # 一个epoch的损失
            train_loss_list.append(train_loss)  # 记录损失
            print(f"epoch:{epoch}------loss:{round(train_loss, 4)}")

        self.save()

    def test(self):
        self.model.eval()
        self.register_hook()
        input_video, _ = next(iter(self.test_dataloader))
        cv2.destroyAllWindows()

        for input_img in input_video.squeeze(0):
            # Play the video
            salient_map = self.get_salient_map(input_img, transform=False)
            cv2.imshow('Video', salient_map)
            cv2.waitKey(20)

        cv2.destroyAllWindows()

    def register_hook(self):
        def hook_fn(module, input, output):
            # 保存ReLU激活后的特征图
            self.feature_maps.append(output)

        # 注册hook
        self.hooks = []  # 用于保存钩子句柄
        self.feature_maps = []
        for layer in self.model.cnn.cnn:
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(hook_fn)

    def remove_hooks(self):
        # 遍历所有句柄并移除钩子
        for hook in self.hooks:
            hook.remove()
        # 清空句柄列表
        self.hooks = []

    def get_salient_map(self, rgb_img, transform=True, combine=True):
        """
        if transform=True, rgb_img: (240, 320, 3), np.uint8
        if transform=False, rgb_img: (3, 240, 320), torch.float32
        """

        def visual_backprop(x):  # x: (3, 240, 320), torch.float32
            self.feature_maps = []

            # 前向传播，获取特征图
            self.model.cnn.cnn(x.unsqueeze(0).to(self.device))

            # 反转特征图列表，以便从最深层开始处理
            self.feature_maps.reverse()

            # 初始化遮罩为最后一个特征图的平均值
            mask = torch.mean(self.feature_maps[0], dim=1, keepdim=True)
            for i in range(1, len(self.feature_maps)):
                # 上采样遮罩
                mask = F.interpolate(mask, size=self.feature_maps[i].size()[2:], mode='nearest')

                # 乘以当前层的平均特征图
                mask *= torch.mean(self.feature_maps[i], dim=1, keepdim=True)

            # 上采样遮罩到原始输入图像大小
            mask = F.interpolate(mask, size=x.size()[1:], mode='bilinear', align_corners=False)

            # 归一化遮罩
            mask = (mask - mask.min()) / (mask.max() - mask.min())

            return mask.squeeze()

        if transform:
            input_img = torch.tensor(rgb_img.astype(np.float32) / 255.).permute((2, 0, 1))  # (3, 240, 320)
        else:
            input_img = rgb_img
        mask = visual_backprop(input_img)
        # 现在，mask包含了输入图像每个像素的贡献度

        mask_np = mask.cpu().detach().numpy()  # 将形状转换为 (240, 320)

        if transform:
            image_np = rgb_img  # (240, 320, 3)
        else:
            image_np = (rgb_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # 将形状转换为 (240, 320, 3)

        mask_np = (mask_np * 255).astype(np.uint8)
        # image_np = (image_np * 255).astype(np.uint8)

        # 将遮罩转换为彩色图像以便可视化
        mask_color = cv2.applyColorMap(mask_np, cv2.COLORMAP_JET)

        if combine:
            # 垂直堆叠原始图像和遮罩
            combined_image = cv2.vconcat([cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), mask_color])
            return combined_image  # BGR format
        else:
            return mask_color


if __name__ == '__main__':
    base_model = Model()
    base_model.load()
    # base_model.train(10)
    base_model.test()
