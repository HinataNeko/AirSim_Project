import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time
import numpy as np

from EnvWrapper_Image import DroneEnvWrapper

MODE = 'train'

input_channels = 3
input_size = (240, 320)  # 目标图像尺寸
state_dim = (input_channels, *input_size)
action_dim = 4

LR_A = 0.0001  # learning rate for actor
LR_C = 0.0001  # learning rate for critic
GAMMA = 0.99  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 20000  # size of replay buffer
BATCH_SIZE = 256  # update batchsize

MAX_EPISODES = 100  # total number of episodes for training
MAX_EP_STEPS = 1000  # total number of steps for each episode
# TEST_PER_EPISODES = 10  # test the model per episodes
VAR = 0.8  # initial exploration variance
VAR_MIN = 0.1  # minimum exploration variance
DECAY_RATE = 0.995  # decay rate per episode


class ReplayBuffer:
    """
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    """

    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity  # buffer的最大值
        self.memory = {
            'state': np.zeros((capacity, *state_dim), dtype=np.float32),
            'action': np.zeros((capacity, action_dim), dtype=np.float32),
            'reward': np.zeros((capacity, 1), dtype=np.float32),
            'next_state': np.zeros((capacity, *state_dim), dtype=np.float32),
        }
        self.position = 0  # 当前输入的位置，相当于指针
        self.is_full = False
        self.memory_save_path = 'saved_model/'
        self.memory_path = self.memory_save_path + 'ddpg_memory.npz'

    def push(self, state, action, reward, next_state):
        self.memory['state'][self.position] = state
        self.memory['action'][self.position] = action
        self.memory['reward'][self.position] = reward
        self.memory['next_state'][self.position] = next_state
        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.is_full = True

    def sample(self, batch_size):
        if batch_size > self.position and not self.is_full:
            raise ValueError("Not enough elements in the buffer to sample")

        if self.is_full:
            indices = np.random.choice(self.capacity, batch_size, replace=False)
        else:
            indices = np.random.choice(self.position, batch_size, replace=False)

        batch = {k: v[indices] for k, v in self.memory.items()}
        return batch['state'], batch['action'], batch['reward'], batch['next_state']

    def save_memory(self):
        if not os.path.exists(self.memory_save_path):
            os.makedirs(self.memory_save_path)

        np.savez(self.memory_path, **self.memory, position=self.position, is_full=self.is_full)
        print('Memory saved!')

    def load_memory(self):
        if os.path.exists(self.memory_path):
            data = np.load(self.memory_path)
            previous_capacity = len(data['state'])
            previous_position = int(data['position'])
            previous_is_full = bool(data['is_full'])

            if previous_capacity > self.capacity:
                if previous_is_full:
                    # 如果之前的缓冲区已满，只保留当前的capacity个最新的数据
                    discard_amount = previous_capacity - self.capacity
                    for k in self.memory.keys():
                        self.memory[k] = data[k][-self.capacity:]
                    self.position = 0
                    self.is_full = True
                else:
                    # 如果之前的缓冲区未满
                    discard_amount = max(previous_position - self.capacity, 0)
                    num_to_keep = min(previous_position, self.capacity)
                    for k in self.memory.keys():
                        self.memory[k][:num_to_keep] = data[k][:num_to_keep]
                    self.position = 0 if previous_position >= self.capacity else num_to_keep
                    self.is_full = previous_position >= self.capacity
                print(f"Warning: Previous memory is larger than current capacity. "
                      f"Discarding the oldest {discard_amount} entries.")
            else:
                for k in self.memory.keys():
                    self.memory[k][:previous_capacity] = data[k]
                self.is_full = previous_capacity == self.capacity and previous_is_full
                self.position = previous_position if previous_capacity == self.capacity else previous_capacity if previous_is_full else previous_position

            print(f'Previous memory loaded! Position: {self.position}, Is full: {self.is_full}')

    def __len__(self):
        return self.capacity if self.is_full else self.position


class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.cnn_layer = nn.Sequential(  # 240*320
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # 120*160
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # 60*80
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # 30*40
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # 15*20
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # mean: (-1, 1)
        )

        self._init_weights()

    def _init_weights(self):
        # Apply He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, state):
        conv_out = self.cnn_layer(state)
        normal_action = self.fc_layer(conv_out)
        return normal_action


class Critic(nn.Module):
    def __init__(self, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.cnn_layer = nn.Sequential(  # 240*320
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # 120*160
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # 60*80
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # 30*40
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # 15*20
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(256 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Apply He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, state, normal_action):
        conv_out = self.cnn_layer(state)
        x = torch.cat([conv_out, normal_action], dim=1)
        out = self.fc_layer(x)
        return out


class DDPG(object):
    def __init__(self, a_dim, s_dim):
        self.memory = ReplayBuffer(MEMORY_CAPACITY, s_dim, a_dim)

        self.learning_started = False
        self.a_dim, self.s_dim = a_dim, s_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create Actor and Critic Networks
        self.actor = Actor(a_dim).to(self.device)
        self.critic = Critic(a_dim).to(self.device)
        self.actor.train()
        self.critic.train()

        # Create target networks
        self.actor_target = Actor(a_dim).to(self.device)
        self.critic_target = Critic(a_dim).to(self.device)

        # Initialize target networks to have same weights as original networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.eval()
        self.critic_target.eval()

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_A)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_C)

        self.model_save_path = 'saved_model/'
        self.model_path = self.model_save_path + 'ddpg.pth'
        self.checkpoint = {'actor_state_dict': self.actor.state_dict(),
                           'actor_target_state_dict': self.actor_target.state_dict(),
                           'critic_state_dict': self.critic.state_dict(),
                           'critic_target_state_dict': self.critic_target.state_dict(),
                           'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                           'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                           }

    def soft_update(self):
        """
        Soft update the target network parameters with source network parameters.
        """
        # Update actor target network
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        # Update critic target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

    # 选择动作，把输入s，输出a
    def choose_action(self, s):  # s: (height, width, channels)
        """
        Choose action
        :param s: state
        :return: act
        """
        # input_s = (s.astype(np.float32) / 255.).transpose((2, 0, 1))
        input_s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        return self.actor(input_s.to(self.device))[0]

    def learn(self):
        # Randomly sample a batch from memory
        batch_s, batch_a, batch_r, batch_s_ = self.memory.sample(BATCH_SIZE)  # 从buffer中sample数据
        batch_r = (batch_r - batch_r.mean()) / (batch_r.std() + 1E-7)
        batch_s = torch.tensor(batch_s, dtype=torch.float32).to(self.device)
        batch_a = torch.tensor(batch_a, dtype=torch.float32).to(self.device)
        batch_r = torch.tensor(batch_r, dtype=torch.float32).to(self.device)
        batch_s_ = torch.tensor(batch_s_, dtype=torch.float32).to(self.device)

        # Critic update
        with torch.no_grad():
            a_ = self.actor_target(batch_s_)
            q_target = self.critic_target(batch_s_, a_)
            y = batch_r + GAMMA * q_target

        q_value = self.critic(batch_s, batch_a)
        critic_loss = F.mse_loss(q_value, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        a = self.actor(batch_s)
        q = self.critic(batch_s, a)
        actor_loss = -torch.mean(q)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update the target networks
        self.soft_update()

    def save_model(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        torch.save(self.checkpoint, self.model_path)
        print('Model saved!')

    def load_model(self):
        if os.path.exists(self.model_path):
            model_data = torch.load(self.model_path)
            self.actor.load_state_dict(model_data['actor_state_dict'])
            self.actor_target.load_state_dict(model_data['actor_target_state_dict'])
            self.critic.load_state_dict(model_data['critic_state_dict'])
            self.critic_target.load_state_dict(model_data['critic_target_state_dict'])
            self.actor_optimizer.load_state_dict(model_data['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(model_data['critic_optimizer_state_dict'])
            print('Previous model loaded!')

    def save_memory(self):
        self.memory.save_memory()

    def load_memory(self):
        self.memory.load_memory()


if __name__ == '__main__':
    env_wrapper = DroneEnvWrapper()
    # env_wrapper.main()

    ddpg = DDPG(action_dim, state_dim)
    ddpg.load_model()
    ddpg.load_memory()

    # 训练部分：
    if MODE == 'train':  # train
        # reward_per_ep = []  # 用于记录每个EP的reward，统计变化
        # t0 = time.time()  # 统计时间
        for i in range(MAX_EPISODES):
            t1 = time.time()
            s = env_wrapper.reset()
            s = (s.astype(np.float32) / 255.).transpose((2, 0, 1))
            for j in range(MAX_EP_STEPS):
                # Add exploration noise
                with torch.no_grad():
                    a = ddpg.choose_action(s).detach().cpu().numpy()

                # 为了能保持开发，这里用了另外一种方式增加探索。
                # 以a为均值，VAR为标准差，建立正态分布，再从正态分布采样出a，然后进行裁剪
                a = np.clip(np.random.normal(a, VAR), -1, 1)

                # 与环境进行互动
                s_, r, done = env_wrapper.step(a)

                # 保存s，a，r，s_
                s_ = (s_.astype(np.float32) / 255.).transpose((2, 0, 1))
                ddpg.memory.push(s, a, r, s_)

                # 开始学习
                if not ddpg.learning_started and len(ddpg.memory) > BATCH_SIZE:
                    ddpg.learning_started = True
                    print(f"Learning started! MEMORY_CAPACITY: {MEMORY_CAPACITY}")  # 提示信息

                if ddpg.learning_started:
                    ddpg.learn()

                s = s_
                if done:
                    break
            print('\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'
                  .format(i, MAX_EPISODES, env_wrapper.episode_reward, time.time() - t1))

            # Decay the exploration variance after each episode
            # VAR = max(VAR * DECAY_RATE, VAR_MIN)

        ddpg.save_model()
        ddpg.save_memory()

    # test
    if MODE == 'test':
        while True:
            s = env_wrapper.reset()
            for i in range(MAX_EP_STEPS):
                with torch.no_grad():
                    a = ddpg.choose_action(s).detach().cpu().numpy()
                a = np.clip(np.random.normal(a, 0.2), -1, 1)
                s, r, done = env_wrapper.step(a)
                if done:
                    break
            print('Episode Reward: {:.4f}'.format(env_wrapper.episode_reward))
