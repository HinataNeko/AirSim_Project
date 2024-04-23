import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly import visualizing
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer, monitor

import time
import os
import copy
import matplotlib.pyplot as plt
import numpy as np

from EnvWrapper_Simple_CNN_RNN import DroneEnvWrapper

NAME = 'TD3_Simple_CNN_SRNN'
MODE = 'train'

state_dim = 128
state_keep_n = 3
action_dim = 4

LR_A = 0.00003  # learning rate for actor
LR_C = 0.00003  # learning rate for critic
GAMMA = 0.99  # reward discount
TAU = 0.005  # soft replacement

MEMORY_CAPACITY = 1000000  # size of replay buffer
BATCH_SIZE = 256  # update batch_size

MAX_EPISODES = 1000  # total number of episodes for training
MAX_EP_STEPS = 400  # total number of steps for each episode
EXPLORE_EPISODES = 0  # for random action sampling in the beginning of training
UPDATE_ITR = 2  # repeated updates for each step
POLICY_UPDATE_FREQ = 4  # 策略网络更新频率

EXPLORE_NOISE_SCALE = 0.35  # range of action noise for exploration
EVAL_NOISE_SCALE = 0.5  # range of action noise for evaluation of action value
NOISE_CLIP = 0.75


class ReplayBuffer:
    """
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    """

    def __init__(self, capacity, state_dim, action_dim, save_folder='./saved_model', file_name=f'{NAME}_memory.npz'):
        self.capacity = capacity  # buffer的最大值
        self.memory = {
            'state': np.zeros((capacity, state_keep_n, state_dim), dtype=np.float32),
            'action': np.zeros((capacity, action_dim), dtype=np.float32),
            'reward': np.zeros((capacity, 1), dtype=np.float32),
            'next_state': np.zeros((capacity, state_keep_n, state_dim), dtype=np.float32),
            'done': np.zeros((capacity, 1), dtype=np.bool_)
        }
        self.position = 0  # 当前输入的位置，相当于指针
        self.is_full = False
        self.memory_save_folder = save_folder
        self.memory_save_path = os.path.join(self.memory_save_folder, file_name)

    def push(self, state, action, reward, next_state, done):
        self.memory['state'][self.position] = state
        self.memory['action'][self.position] = action
        self.memory['reward'][self.position] = reward
        self.memory['next_state'][self.position] = next_state
        self.memory['done'][self.position] = done
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
        return batch['state'], batch['action'], batch['reward'], batch['next_state'], batch['done']

    def save_memory(self):
        if not os.path.exists(self.memory_save_folder):
            os.makedirs(self.memory_save_folder)

        np.savez(self.memory_save_path, **self.memory, position=self.position, is_full=self.is_full)
        print('Memory saved!')

    def load_memory(self):
        if os.path.exists(self.memory_save_path):
            data = np.load(self.memory_save_path)
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


class DoubleReplayBuffer:
    """
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    """

    def __init__(self, capacity, state_dim, action_dim, save_folder='./saved_model'):
        self.good_replay_buffer = ReplayBuffer(capacity // 2, state_dim, action_dim, save_folder=save_folder,
                                               file_name=f'{NAME}_good_memory.npz')
        self.bad_replay_buffer = ReplayBuffer(capacity // 2, state_dim, action_dim, save_folder=save_folder,
                                              file_name=f'{NAME}_bad_memory.npz')

    def push(self, state, action, reward, next_state, done):
        if reward >= 0:
            self.good_replay_buffer.push(state, action, reward, next_state, done)
        else:
            self.bad_replay_buffer.push(state, action, reward, next_state, done)

    def sample(self, batch_size):
        good_batch = bad_batch = None
        if len(self.good_replay_buffer) >= batch_size // 2:
            good_batch = self.good_replay_buffer.sample(batch_size // 2)
        if len(self.bad_replay_buffer) >= batch_size // 2:
            bad_batch = self.bad_replay_buffer.sample(batch_size // 2)

        if good_batch is None:
            return [np.concatenate([batch, batch], axis=0) for batch in bad_batch]
        elif bad_batch is None:
            return [np.concatenate([batch, batch], axis=0) for batch in good_batch]
        return [np.concatenate([batch1, batch2], axis=0) for batch1, batch2 in zip(good_batch, bad_batch)]

    def save_memory(self):
        self.good_replay_buffer.save_memory()
        self.bad_replay_buffer.save_memory()

    def load_memory(self):
        self.good_replay_buffer.load_memory()
        self.bad_replay_buffer.load_memory()

    def __len__(self):
        return len(self.good_replay_buffer) + len(self.bad_replay_buffer)


class SpikingRNNBase(nn.Module):
    def __init__(self, input_size, hidden_size, v_threshold: float = 1.):
        super(SpikingRNNBase, self).__init__()
        self.i2h = layer.Linear(input_size, hidden_size, step_mode='m')
        self.h2h = layer.Linear(hidden_size, hidden_size, step_mode='m')
        self.lif_layer = neuron.LIFNode(v_threshold=v_threshold, step_mode='m')

    def forward(self, input, hidden):  # (T, batch_size, hidden)
        # 输入到隐藏状态的线性变换
        input_hidden = self.i2h(input)
        # 隐藏状态到隐藏状态的线性变换
        hidden_hidden = self.h2h(hidden)
        # 应用激活函数并更新隐藏状态
        next_hidden = self.lif_layer(input_hidden + hidden_hidden)
        return next_hidden


class SpikingRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SpikingRNN, self).__init__()
        self.rnn1 = SpikingRNNBase(input_size, hidden_size)
        self.rnn2 = SpikingRNNBase(hidden_size, hidden_size, v_threshold=999.)

    def forward(self, input, hidden):
        """
        input: (T, batch_size, frames, hidden)
        hidden: (num_layers, T, batch_size, hidden)
        """
        frames = input.shape[2]

        h1 = hidden[0]  # (T, batch_size, hidden)
        rnn1_hidden_list = []
        for frame in range(frames):
            h1 = self.rnn1(input[:, :, frame], h1)  # (T, batch_size, hidden)
            rnn1_hidden_list.append(h1)
            functional.reset_net(self.rnn1)

        rnn1_output_hidden = torch.stack(rnn1_hidden_list, dim=2)  # (T, batch_size, frames, hidden)

        h2 = hidden[1]  # (T, batch_size, hidden)
        v_list = []
        for frame in range(frames):
            h2 = self.rnn2(rnn1_output_hidden[:, :, frame], h2)  # (T, batch_size, hidden)
            v_list.append(self.rnn2.lif_layer.v)  # 最后一层膜电位，(batch_size, hidden)
            functional.reset_net(self.rnn2)

        rnn_output_v = torch.stack(v_list, dim=1)  # (batch_size, frames, hidden)
        return rnn_output_v, torch.stack([h1, h2], dim=0)


class QNetwork(nn.Module):
    """ the network for evaluating values of state-action pairs: Q(s,a) in PyTorch using nn.Sequential """

    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = 2
        self.T = 8

        self.srnn = SpikingRNN(input_size=self.state_dim, hidden_size=self.hidden_dim)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        self._init_weights(init_w)

    def _init_weights(self, init_w):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -init_w, init_w)
                nn.init.uniform_(m.bias, -init_w, init_w)

    def forward(self, state, action):
        current_batch_size = state.shape[0]
        rnn_input = state.unsqueeze(0).repeat(self.T, *([1] * state.dim()))  # (T, batch_size, frames, hidden)
        h_0 = torch.zeros(self.rnn_layers, self.T, current_batch_size, self.hidden_dim).to(self.device)
        rnn_output, _ = self.srnn(rnn_input, h_0)  # rnn_output: (batch_size, time_seq, hidden_size)

        x = torch.cat([rnn_output[:, -1], action], dim=1)
        return self.fc(x)


class ActorNetwork(nn.Module):
    """ the network for generating non-deterministic (Gaussian distributed) action from the state input in PyTorch """

    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3):
        super(ActorNetwork, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = 2
        self.T = 8

        self.srnn = SpikingRNN(input_size=self.state_dim, hidden_size=self.hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self._init_weights(init_w)

    def _init_weights(self, init_w):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -init_w, init_w)
                nn.init.uniform_(m.bias, -init_w, init_w)

    def forward(self, state):
        current_batch_size = state.shape[0]
        rnn_input = state.unsqueeze(0).repeat(self.T, *([1] * state.dim()))  # (T, batch_size, frames, hidden)
        h_0 = torch.zeros(self.rnn_layers, self.T, current_batch_size, self.hidden_dim).to(self.device)
        rnn_output, _ = self.srnn(rnn_input, h_0)  # rnn_output: (batch_size, time_seq, hidden_size)

        normal_action = self.fc(rnn_output[:, -1])
        return normal_action

    def evaluate(self, state, eval_noise_scale):
        """
        generate action with state for calculating gradients;
        eval_noise_scale: as the trick of target policy smoothing, for generating noisy actions.
        """
        # state = torch.tensor(state, dtype=torch.float32)
        normal_action = self.forward(state)

        # add noise
        noise = (torch.randn_like(normal_action) * eval_noise_scale).clamp(-NOISE_CLIP, NOISE_CLIP).to(self.device)
        normal_action = (normal_action + noise).clamp(-1, 1)
        return normal_action

    # 输入single state，输出action
    def get_action(self, state, explore_noise_scale):
        """ generate action with state for interaction with environment """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # add batch dimension
        normal_action = self.forward(state.to(self.device))
        normal_action = normal_action.squeeze(0).detach().cpu()  # remove batch dimension

        # add noise
        noise = (torch.randn_like(normal_action) * explore_noise_scale).clamp(-NOISE_CLIP, NOISE_CLIP)
        normal_action = (normal_action + noise).clamp(-1, 1)
        return normal_action.numpy()

    def random_sample_action(self):
        """ generate random actions for exploration """
        random_normal_action = torch.rand(self.action_dim) * 2 - 1  # 均匀分布，通过缩放和平移来匹配[-1, 1]区间
        return random_normal_action.numpy()


class TD3:
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=256,
                 ):
        self.save_folder = './saved_model/'
        self.model_save_path = os.path.join(self.save_folder, f'{NAME}_model.pth')
        self.replay_buffer = DoubleReplayBuffer(MEMORY_CAPACITY, state_dim, action_dim, save_folder=self.save_folder)

        self.update_cnt = 0  # 更新次数
        self.learning_started = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # initialize all networks
        # 用两个Qnet来估算，doubleDQN的想法。同时也有两个对应的target_q_net
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q_net1_target = copy.deepcopy(self.q_net1)
        self.q_net2_target = copy.deepcopy(self.q_net2)
        self.actor_target = copy.deepcopy(self.actor)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_A)
        self.q_optimizer1 = torch.optim.Adam(self.q_net1.parameters(), lr=LR_C)
        self.q_optimizer2 = torch.optim.Adam(self.q_net2.parameters(), lr=LR_C)

        self.checkpoint = {'actor_state_dict': self.actor.state_dict(),
                           'actor_target_state_dict': self.actor_target.state_dict(),
                           'q_net1_state_dict': self.q_net1.state_dict(),
                           'q_net1_target_state_dict': self.q_net1_target.state_dict(),
                           'q_net2_state_dict': self.q_net2.state_dict(),
                           'q_net2_target_state_dict': self.q_net2_target.state_dict(),
                           'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                           'q_optimizer1_state_dict': self.q_optimizer1.state_dict(),
                           'q_optimizer2_state_dict': self.q_optimizer2.state_dict(),
                           }

    # 软更新
    def target_soft_update(self):
        """ Soft update the target network parameters with source network parameters """
        # Update actor target network
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        # Update critic target network
        for target_param, param in zip(self.q_net1_target.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        # Update critic target network
        for target_param, param in zip(self.q_net2_target.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

    def update(self):
        """ update all networks in TD3 """
        self.update_cnt += 1  # 计算更新次数
        state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)  # 从buffer中sample数据

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # 输入s',从target_policy_net计算a'。注意这里有加noisy的
            # clipped normal noise
            new_next_action = self.actor_target.evaluate(next_state, EVAL_NOISE_SCALE)

            # 归一化reward.(有正有负)
            # normalize with batch mean and std; plus a small number to prevent numerical problem
            # reward = self.reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)

            # Training Q Function
            # 把s'和a'堆叠在一起，一起输入到target_q_net。
            # 有两个qnet，我们取最小值
            target_q_min = torch.min(self.q_net1_target(next_state, new_next_action),
                                     self.q_net2_target(next_state, new_next_action))

            # 计算target_q的值，用于更新q_net
            # 之前把done从布尔变量改为int，就是为了这里能够直接计算。
            target_q_value = reward + (1 - done) * GAMMA * target_q_min  # if done==1, only reward

        # Update Q-net1 and Q-net2
        predicted_q_value1 = self.q_net1(state, action)
        predicted_q_value2 = self.q_net2(state, action)
        q_value_loss1 = F.mse_loss(predicted_q_value1, target_q_value)
        q_value_loss2 = F.mse_loss(predicted_q_value2, target_q_value)
        q_value_loss = q_value_loss1 + q_value_loss2
        self.q_optimizer1.zero_grad()
        self.q_optimizer2.zero_grad()
        q_value_loss.backward()
        self.q_optimizer1.step()
        self.q_optimizer2.step()

        # Training Policy Function
        # policy不是经常update的，而是qnet更新一定次数，才update一次
        if self.update_cnt % POLICY_UPDATE_FREQ == 0:
            new_action = self.actor.evaluate(state, eval_noise_scale=0.0)
            predicted_new_q_value = self.q_net1(state, new_action)
            actor_loss = -predicted_new_q_value.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update the three target nets
            self.target_soft_update()

    def save_weights(self):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        torch.save(self.checkpoint, self.model_save_path)
        print('Model saved!')

    def load_weights(self):  # load trained weights
        if os.path.exists(self.model_save_path):
            model_data = torch.load(self.model_save_path)
            self.actor.load_state_dict(model_data['actor_state_dict'])
            self.actor_target.load_state_dict(model_data['actor_target_state_dict'])
            self.q_net1.load_state_dict(model_data['q_net1_state_dict'])
            self.q_net1_target.load_state_dict(model_data['q_net1_target_state_dict'])
            self.q_net2.load_state_dict(model_data['q_net2_state_dict'])
            self.q_net2_target.load_state_dict(model_data['q_net2_target_state_dict'])
            self.actor_optimizer.load_state_dict(model_data['actor_optimizer_state_dict'])
            self.q_optimizer1.load_state_dict(model_data['q_optimizer1_state_dict'])
            self.q_optimizer2.load_state_dict(model_data['q_optimizer2_state_dict'])
            print('Previous model loaded!')

    def save_memory(self):
        self.replay_buffer.save_memory()

    def load_memory(self):
        self.replay_buffer.load_memory()


if __name__ == '__main__':
    env_wrapper = DroneEnvWrapper(render=True)

    # initialization of trainer
    td3 = TD3(state_dim=state_dim, action_dim=action_dim)

    td3.load_weights()
    td3.load_memory()

    # training loop
    if MODE == 'train':  # train
        td3.q_net1.train()
        td3.q_net2.train()
        td3.actor.train()
        td3.q_net1_target.eval()
        td3.q_net2_target.eval()
        td3.actor_target.eval()

        n = 0
        reward_per_ep = []  # 用于记录每个EP的reward，统计变化
        while n < MAX_EPISODES:
            t1 = time.time()
            step = 0
            state = env_wrapper.reset()  # 初始化
            if state is None:
                continue

            # an episode
            for j in range(MAX_EP_STEPS):
                if n >= EXPLORE_EPISODES:
                    # 带有noisy的action
                    action = td3.actor.get_action(state, explore_noise_scale=EXPLORE_NOISE_SCALE)
                else:  # 随机
                    action = td3.actor.random_sample_action()

                # 与环境进行交互
                next_state, reward, done, successful = env_wrapper.step(action)

                # 异常episode
                if j < 15 and done:
                    break

                # 记录数据在replay_buffer
                td3.replay_buffer.push(state, action, reward, next_state, done)

                # 如果数据超过一个batch_size的大小，那么就开始学习
                if not td3.learning_started and len(td3.replay_buffer) > BATCH_SIZE:
                    td3.learning_started = True
                    print(f"Learning started! MEMORY_CAPACITY: {MEMORY_CAPACITY}")  # 提示信息

                if td3.learning_started:
                    for _ in range(UPDATE_ITR):  # 可以更新多次
                        td3.update()

                state = next_state
                step += 1
                if done:
                    break

            # 异常episode
            if step < 15:
                continue

            n += 1
            reward_per_ep.append(env_wrapper.episode_reward)
            print('\rEpisode: {}/{} | '
                  'Episode Reward: {:.4f} | '
                  'Distance Reward: {:.4f} | '
                  'Detection Reward: {:.4f} | '
                  'Final Reward: {:.4f} | '
                  'Step: {} | '
                  'Running Time: {:.2f} | '
                  'Good Buffer: {} | '
                  'Bad Buffer: {}'
                  .format(n, MAX_EPISODES,
                          env_wrapper.episode_reward,
                          env_wrapper.episode_distance_reward,
                          env_wrapper.episode_detection_reward,
                          env_wrapper.episode_final_reward,
                          step, time.time() - t1,
                          len(td3.replay_buffer.good_replay_buffer), len(td3.replay_buffer.bad_replay_buffer)))
            if n != 0 and n % 200 == 0:
                td3.save_weights()
                td3.save_memory()

        np.save(f"./saved_model/{NAME}_reward_history.npy", np.array(reward_per_ep))
        plt.plot(reward_per_ep)
        plt.show()

        td3.save_weights()
        td3.save_memory()

    elif MODE == 'test':  # test
        td3.q_net1.eval()
        td3.q_net2.eval()
        td3.actor.eval()
        td3.q_net1_target.eval()
        td3.q_net2_target.eval()
        td3.actor_target.eval()

        n_test = 100
        n_successful = 0
        n = 0

        while n < n_test:
            t1 = time.time()
            step = 0
            state = env_wrapper.reset()
            if state is None:
                continue

            for _ in range(2 * MAX_EP_STEPS):
                action = td3.actor.get_action(state, explore_noise_scale=0)
                next_state, reward, done, successful = env_wrapper.step(action)

                state = next_state
                step += 1
                if done:
                    if successful:
                        n_successful += 1
                    break

            # 异常episode

            if step < 15:
                continue

            n += 1
            print('\rEpisode: {} | '
                  'Episode Reward: {:.4f} | '
                  'Distance Reward: {:.4f} | '
                  'Detection Reward: {:.4f} | '
                  'Final Reward: {:.4f} | '
                  'Step: {} | '
                  'Running Time: {:.2f}'
                  .format(n,
                          env_wrapper.episode_reward,
                          env_wrapper.episode_distance_reward,
                          env_wrapper.episode_detection_reward,
                          env_wrapper.episode_final_reward,
                          step, time.time() - t1))

        print(f"Successful rate: {n_successful}/{n_test}={n_successful / n_test}")
