import matplotlib.pyplot as plt
import numpy as np


def plot_srnn_hyperparameter_effects_on_success_rate():
    # 数据
    T_values = [4, 6, 8]
    success_rates = {
        'H=32': [56, 66, 71],
        'H=64': [71, 78, 81],
        'H=128': [78, 84, 86]
    }

    # 设置图表样式
    plt.figure(figsize=(8, 6))
    fontsize = 18
    markers = ['o', 's', '^']  # 不同折线图的标记
    colors = ['blue', 'orange', 'green']

    # 绘制每条折线
    for (legend, rates), marker, color in zip(success_rates.items(), markers, colors):
        plt.plot(T_values, rates, marker=marker, linestyle='-', linewidth=3, color=color, label=legend, markersize=12)

    # 设置图表标题和轴标签
    plt.title('Effect of Hyperparameters on Navigation Success Rate', fontsize=fontsize)
    plt.xlabel('Time Steps (T)', fontsize=fontsize)
    plt.ylabel('Success Rate (%)', fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.legend(fontsize=fontsize, frameon=False, loc='lower right')
    # plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)  # 设置网格线 - 只显示横向
    plt.grid(False)
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
    plt.figure(figsize=(8, 6))
    fontsize = 18
    markers = ['o', 's', '^', 'v', 'D']  # 不同折线图的标记
    colors = ['blue', 'orange', 'green', 'red', 'purple']

    # 绘制每条折线
    for (legend, rates), marker, color in zip(success_rates.items(), markers, colors):
        plt.plot(distances, rates, marker=marker, linestyle='-', linewidth=3, color=color, label=legend, markersize=12)

    plt.title('Navigation Success Rate by Initial Distance across Different Scenes', fontsize=fontsize)
    plt.xlabel('Initial Distance (m)', fontsize=fontsize)
    plt.ylabel('Success Rate (%)', fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.legend(fontsize=fontsize, frameon=False)
    # plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)  # 设置网格线 - 只显示横向
    plt.grid(False)
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
    plt.figure(figsize=(8, 6))
    fontsize = 18
    markers = ['o', 's', '^', 'D']  # 不同折线图的标记
    colors = ['blue', 'orange', 'green', 'red']

    # 绘制每条折线
    for (legend, rates), marker, color in zip(success_rates.items(), markers, colors):
        plt.plot(distances, rates, marker=marker, linestyle='-', linewidth=3, color=color, label=legend, markersize=12)

    # 标识训练场景下的初始距离范围（9~14米）
    # plt.axvspan(9, 14, color='grey', alpha=0.3)
    # plt.axvline(x=14, color='grey', linestyle='--', alpha=0.7)  # 添加虚线边界

    plt.title('Navigation Success Rate by Initial Distance across Different Models', fontsize=fontsize)
    plt.xlabel('Initial Distance (m)', fontsize=fontsize)
    plt.ylabel('Success Rate (%)', fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.legend(fontsize=fontsize, frameon=False)  # 去掉图例的边界框
    # plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)  # 设置网格线 - 只显示横向
    plt.grid(False)
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
    plt.figure(figsize=(8, 6))
    fontsize = 18
    markers = ['o', 's', '^', 'D']  # 不同折线图的标记
    colors = ['blue', 'orange', 'green', 'red']

    # 绘制每条折线
    for (legend, rates), marker, color in zip(success_rates.items(), markers, colors):
        plt.plot(distances, rates, marker=marker, linestyle='-', linewidth=3, color=color, label=legend, markersize=12)

    plt.title('Navigation Success Rate by Noise Var across Different Models', fontsize=fontsize)
    plt.xlabel('Input Noise Variance', fontsize=fontsize)
    plt.ylabel('Success Rate (%)', fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.legend(fontsize=fontsize, frameon=False)  # 去掉图例的边界框
    # plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)  # 设置网格线 - 只显示横向
    plt.grid(False)
    plt.xticks(distances)
    plt.ylim(40, 102)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_srnn_hyperparameter_effects_on_success_rate()
    # plot_srnn_navigation_success_rate_by_distance()
    # plot_navigation_success_rate_by_distance_on_different_models()
    plot_navigation_success_rate_by_noise_on_different_models()
