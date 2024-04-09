import matplotlib.pyplot as plt
import numpy as np


def plot_srnn_hyperparameter_effects_on_success_rate():
    # 数据
    T_values = [4, 6, 8]
    success_rates = {
        32: [56, 66, 71],
        64: [71, 78, 81],
        128: [78, 84, 86]
    }

    # 定义不同H值对应的标记点
    markers = {32: 'o', 64: 's', 128: '^'}  # 'o': 圆圈, 's': 正方形, '^': 上三角

    # 创建图表
    plt.figure(figsize=(5, 4))

    # 绘制每条折线
    for H, rates in success_rates.items():
        plt.plot(T_values, rates, marker=markers[H], linestyle='-', linewidth=2, label=f'H={H}')

    # 设置图表标题和轴标签
    plt.title('Effect of Hyperparameters on Navigation Success Rate', fontsize=14)
    plt.xlabel('Time Steps (T)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.legend(title='Hidden Neurons (H)', fontsize=10)
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)  # 设置网格线 - 只显示横向
    plt.xticks(T_values)
    plt.ylim(50, 90)

    plt.tight_layout()
    plt.show()


def plot_srnn_navigation_success_rate_by_distance():
    # 数据
    distances = [10, 15, 20, 25]  # 初始距离
    success_rates = {
        'Grassland Stadium': [100, 99.33, 98, 87.33],
        'Fountain Park': [100, 98, 89.33, 82],
        'Highway': [99.33, 92, 85.33, 62.67],
        'Tennis Court': [98.67, 85.33, 74, 56],
        'Lakeside': [98, 80, 60, 53.33]
    }

    # 设置图表样式
    plt.figure(figsize=(5, 4))
    markers = ['o', 's', '^', 'D', '*']  # 不同场景的标记
    colors = plt.cm.viridis(np.linspace(1, 0, len(success_rates)))  # 颜色渐变

    # 绘制每条折线
    for (scene, rates), marker, color in zip(success_rates.items(), markers, colors):
        plt.plot(distances, rates, marker=marker, linestyle='-', linewidth=2, color=color, label=scene)

    # 标识训练场景下的初始距离范围（9~14米）
    # plt.axvspan(9, 14, color='grey', alpha=0.3)
    # plt.axvline(x=14, color='grey', linestyle='--', alpha=0.7)  # 添加虚线边界

    plt.title('Navigation Success Rate by Initial Distance across Different Scenes', fontsize=14)
    plt.xlabel('Initial Distance (m)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.legend(title='Navigation Scene', fontsize=10)
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)  # 设置网格线 - 只显示横向
    plt.xticks(distances)
    plt.ylim(50, 105)

    plt.tight_layout()
    plt.show()


def plot_navigation_success_rate_by_distance_on_different_models():
    # 数据
    distances = [10, 15, 20, 25]  # 初始距离
    success_rates = {
        'CNN-NaiveRNN': [100, 85.33, 45, 36],
        'CNN-LSTM': [98.67, 82, 54.66, 42],
        'CNN-GRU': [96.67, 79.33, 47.33, 36.66],
        'CNN-SRNN': [100, 99.33, 98, 87.33],
    }

    # 设置图表样式
    plt.figure(figsize=(5, 4))
    markers = ['o', 's', '^', 'D']  # 不同场景的标记
    colors = plt.cm.viridis(np.linspace(1, 0, len(success_rates)))  # 颜色渐变

    # 绘制每条折线
    for (model, rates), marker, color in zip(success_rates.items(), markers, colors):
        plt.plot(distances, rates, marker=marker, linestyle='-', linewidth=2, color=color, label=model)

    # 标识训练场景下的初始距离范围（9~14米）
    # plt.axvspan(9, 14, color='grey', alpha=0.3)
    # plt.axvline(x=14, color='grey', linestyle='--', alpha=0.7)  # 添加虚线边界

    plt.title('Navigation Success Rate by Initial Distance across Different Models', fontsize=14)
    plt.xlabel('Initial Distance (m)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.legend(title='Models', fontsize=10)
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)  # 设置网格线 - 只显示横向
    plt.xticks(distances)
    plt.ylim(30, 102)

    plt.tight_layout()
    plt.show()


def plot_navigation_success_rate_by_noise_on_different_models():
    # 数据
    distances = [0, 0.2, 0.3, 0.4, 0.5]  # 初始距离
    success_rates = {
        'CNN-NaiveRNN': [98, 94, 90, 79, 61],
        'CNN-LSTM': [97, 95, 92, 76, 64],
        'CNN-GRU': [96, 92, 89, 74, 59],
        'CNN-SRNN': [99, 97, 96, 92, 85],
    }

    # 设置图表样式
    plt.figure(figsize=(5, 4))
    markers = ['o', 's', '^', 'D']  # 不同标记
    colors = plt.cm.viridis(np.linspace(1, 0, len(success_rates)))  # 颜色渐变

    # 绘制每条折线
    for (model, rates), marker, color in zip(success_rates.items(), markers, colors):
        plt.plot(distances, rates, marker=marker, linestyle='-', linewidth=2, color=color, label=model)

    plt.title('Navigation Success Rate by Noise Var across Different Models', fontsize=14)
    plt.xlabel('Initial Distance (m)', fontsize=12)
    plt.ylabel('Noise Var', fontsize=12)
    plt.legend(title='Models', fontsize=10)
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)  # 设置网格线 - 只显示横向
    plt.xticks(distances)
    plt.ylim(40, 102)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_srnn_hyperparameter_effects_on_success_rate()
    # plot_srnn_navigation_success_rate_by_distance()
    plot_navigation_success_rate_by_noise_on_different_models()
