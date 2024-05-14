import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def smooth(data, sm=5):
    z = np.ones(len(data))
    y = np.ones(sm) * 1.0
    d = np.convolve(y, data, "same") / np.convolve(y, z, "same")
    return d


def exponential_moving_average(rewards, alpha=0.1):
    ema = []
    for i, r in enumerate(rewards):
        if i == 0:
            ema.append(r)
        else:
            ema.append(alpha * r + (1 - alpha) * ema[-1])
    return np.array(ema)


def shift_array(arr, n, direction):
    """
    Shifts an array to the left or right by n units, filling the new values
    with the last (or first) element from the original array.

    Parameters:
        arr (np.ndarray): The input array to shift.
        n (int): The number of positions to shift the array.
        direction (str): 'left' for left shift, 'right' for right shift.

    Returns:
        np.ndarray: The shifted array with filled values.
    """
    if direction == 'left':
        shifted_arr = np.roll(arr, -n)
        shifted_arr[-n:] = arr[-1]
    elif direction == 'right':
        shifted_arr = np.roll(arr, n)
        shifted_arr[:n] = arr[0]
    else:
        raise ValueError("direction must be 'left' or 'right'")

    return shifted_arr


def load_reward_history(file_path):
    data = np.load(file_path)
    total_reward = smooth(data['total_reward'])
    distance_reward = smooth(data['distance_reward'])
    detection_reward = smooth(data['detection_reward'])
    return total_reward, distance_reward, detection_reward


def plot_reward(result, title="Reward per Episode"):
    """
    result: 一维的ndarray。
    """

    plt.figure(figsize=(8, 6))
    episodes = list(range(1, len(result) + 1))

    plt.plot(episodes, result, 'b-', label='Total Reward')  # 使用蓝色实线
    plt.title(title, fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.tick_params(labelsize=12)
    plt.grid(True)

    plt.tight_layout()  # 自动调整子图参数
    plt.show()


def plot_rewards_with_multi_seeds(results):
    """
    results (list of np.ndarray): 每个元素是一个含有特定随机种子训练结果的 ndarray。
    """

    episodes = np.arange(1, len(results[0]) + 1)

    # 创建一个包含所有结果的数据框，每个种子的数据作为一个单独的列
    df = pd.DataFrame(index=episodes)

    # 为每个随机种子结果添加一列
    for i, seed_results in enumerate(results):
        df[f'Seed {i + 1}'] = seed_results

    # 设置图形大小和风格
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # 为每个随机种子绘制曲线
    for column in df.columns:
        sns.lineplot(data=df, x=df.index, y=column, label=column, lw=2)

    # 设置图例、标题和坐标轴标签
    plt.legend(title='Random Seed')
    plt.title('TD3 Training Performance for Multiple Seeds', fontsize=14)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_rewards_with_shade(results, title=''):
    """
    results (list of np.ndarray): 每个元素是一个含有特定随机种子训练结果的 ndarray。
    """

    # 将数据转换为 DataFrame
    episodes = np.arange(1, len(results[0]) + 1)
    df = pd.DataFrame({
        'Episode': np.tile(episodes, len(results)),
        'Reward': np.concatenate(results)
    })

    plt.figure(figsize=(8, 6))
    fontsize = 14
    # sns.set(style="whitegrid")  # 设置 Seaborn 的背景和网格样式为白色网格

    # 绘制带有置信区间的奖励曲线
    sns.lineplot(data=df, x='Episode', y='Reward', errorbar=('ci', 95), color='blue', lw=2)
    plt.title(title, fontsize=fontsize)
    plt.xlabel('Episode', fontsize=fontsize)
    plt.ylabel('Reward', fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.grid(False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    total_reward_list, distance_reward_list, detection_reward_list = [], [], []
    for i in range(1, 10):
        total_reward, distance_reward, detection_reward = load_reward_history(
            f"./saved_model/backup/TD3_Simple_CNN_CfC_reward_history({i}).npz")
        total_reward_list.append(total_reward)
        distance_reward_list.append(distance_reward)
        detection_reward_list.append(detection_reward)

    # plot_reward(total_reward_list[0], title="Total Reward per Episode")
    # plot_reward(distance_reward_list[0], title="Distance Reward per Episode")
    plot_rewards_with_shade(total_reward_list, title='Total Reward per Episode')
    plot_rewards_with_shade(distance_reward_list, title='Distance Reward per Episode')
    # plot_rewards_with_shade(detection_reward_list)
