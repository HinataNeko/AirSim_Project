import matplotlib.pyplot as plt
import numpy as np


def plot_hyperparameter_effects_on_success_rate():
    # 定义数据
    T_values = [4, 6, 8]
    success_rates = {
        32: [56, 66, 71],
        64: [71, 78, 81],
        128: [78, 84, 86]
    }

    # 定义不同H值对应的标记点
    markers = {32: 'o', 64: 's', 128: '^'}  # 'o': 圆圈, 's': 正方形, '^': 上三角

    # 创建图表
    plt.figure(figsize=(8, 6))

    # 绘制每条折线
    for H, rates in success_rates.items():
        plt.plot(T_values, rates, marker=markers[H], linestyle='-', linewidth=2, label=f'H={H}')

    # 设置图表标题和轴标签
    plt.title('Effect of Hyperparameters on Navigation Success Rate', fontsize=14)
    plt.xlabel('Time Steps (T)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)

    # 添加图例
    plt.legend(title='Hidden Neurons (H)', fontsize=10)

    # 设置网格线 - 只显示横向
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)

    # 设置轴范围
    plt.ylim(50, 90)

    # 优化图表布局
    plt.tight_layout()

    # 显示图表
    plt.show()


def plot_navigation_success_rate_by_distance():
    # 导航成功率数据
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
    markers = ['o', 's', '^', 'D', '*']  # 不同场景的标记
    colors = plt.cm.viridis(np.linspace(1, 0, len(success_rates)))  # 颜色渐变

    # 绘制每条折线
    for (scene, rates), marker, color in zip(success_rates.items(), markers, colors):
        plt.plot(distances, rates, marker=marker, linestyle='-', linewidth=2, color=color, label=scene)

    # 标识训练场景下的初始距离范围（9~14米）
    plt.axvspan(9, 14, color='grey', alpha=0.3)
    plt.axvline(x=14, color='grey', linestyle='--', alpha=0.7)  # 添加虚线边界

    # 设置图表标题和轴标签
    plt.title('Navigation Success Rate by Initial Distance across Different Scenes', fontsize=14)
    plt.xlabel('Initial Distance (m)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)

    # 添加图例
    plt.legend(title='Navigation Scene', fontsize=10)

    # 设置网格线 - 只显示横向
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)

    # 设置轴范围
    plt.xticks(distances)
    plt.ylim(50, 105)

    # 优化图表布局
    plt.tight_layout()

    # 显示图表
    plt.show()


if __name__ == '__main__':
    # plot_hyperparameter_effects_on_success_rate()
    plot_navigation_success_rate_by_distance()
