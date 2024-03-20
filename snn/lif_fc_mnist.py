import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from spikingjelly import visualizing
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer, rnn


class SNN(nn.Module):
    def __init__(self, T=64, tau=2., surrogate_function=surrogate.Sigmoid()):
        super().__init__()
        self.T = T

        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 1024, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate_function, store_v_seq=True),
            layer.Linear(1024, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate_function, store_v_seq=True),
        )

    def forward(self, x: torch.Tensor):  # (batch_size, *)
        repeated_x = x.unsqueeze(0).repeat(self.T, *([1] * x.dim()))  # (T, batch_size, *)
        return self.layer(repeated_x)


def train(epochs=20, start_epoch=1):
    """
    使用全连接-LIF的网络结构，进行MNIST识别。
    这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。
    """

    # 初始化数据加载器
    train_dataset = torchvision.datasets.MNIST(
        root='./datasets',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./datasets',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_data_loader = data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=False)
    test_data_loader = data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, drop_last=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'snn_mnist'
    model_save_path = './saved_model'
    T = 8
    net = SNN(T=T, tau=2.).to(device)
    functional.set_step_mode(net, step_mode='m')
    # encoder = encoding.PoissonEncoder()
    optimizer = torch.optim.Adam(net.parameters())

    checkpoint = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    max_test_acc = -1

    # if resume:
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     net.load_state_dict(checkpoint['net'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     max_test_acc = checkpoint['max_test_acc']

    for epoch in range(start_epoch, start_epoch + epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in tqdm(train_data_loader, file=sys.stdout, leave=False):
            img = img.to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, 10).float()

            out = net(img)  # input: (batch_size, channels), output: (T, batch_size, channels)
            out_fr = out.sum(dim=0) / T
            loss = F.mse_loss(out_fr, label_onehot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_samples += label.shape[0]
            train_loss += loss.item() * label.shape[0]
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_loss /= train_samples
        train_acc /= train_samples

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in tqdm(test_data_loader, file=sys.stdout, leave=False):
                img = img.to(device)
                label = label.to(device)
                label_onehot = F.one_hot(label, 10).float()

                out = net(img)  # input: (batch_size, channels), output: (T, batch_size, channels)
                out_fr = out.sum(dim=0) / T
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_loss /= test_samples
        test_acc /= test_samples

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        if save_max:
            torch.save(checkpoint, os.path.join(model_save_path, f'{model_name}_max.pth'))

        print(
            f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')

    torch.save(checkpoint, os.path.join(model_save_path, f'{model_name}_latest.pth'))


if __name__ == '__main__':
    train(epochs=20)
