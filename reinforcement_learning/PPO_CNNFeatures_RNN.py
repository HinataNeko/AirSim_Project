import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from ncps.torch import CfC
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

from EnvWrapper_Simple_CNN_RNN import DroneEnvWrapper

device = torch.device('cuda:0')
NAME = 'PPO_CNNFeatures_CfC'
MODE = 'train'

STATE_DIM_H = 128
STATE_KEEP_N = 3
ACTION_DIM = 4

"""Environment hyperparameters"""
MAX_EP_LEN = 400  # max timesteps in one episode
MAX_TRAINING_TIMESTEPS = int(1000000)  # break training loop if timeteps > max_training_timesteps
MAX_TRAINING_EPISODES = 1000

SAVE_MODEL_FREQ = int(1e5)  # save model frequency (in num timesteps)

ACTION_STD_INIT = 0.6  # starting std for action distribution (Multivariate Normal)
ACTION_STD_DECAY_FREQ = int(1e5)  # action_std decay frequency (in num timesteps)
ACTION_STD_DECAY_RATE = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
MIN_ACTION_STD = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)

"""PPO hyperparameters"""
# UPDATE_TIMESTEP = MAX_EP_LEN * 4  # update policy every n timesteps
UPDATE_TIMESTEP = 4096
K_EPOCHS = 512  # update policy for K epochs in one PPO update

EPS_CLIP = 0.2  # clip parameter for PPO
GAMMA = 0.99  # discount factor

LR_ACTOR = 0.0003  # learning rate for actor network
LR_CRITIC = 0.001  # learning rate for critic network


class Buffer:
    def __init__(self):
        self.transition_dict = {
            'states': [],
            'actions': [],
            'action_logprobs': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }

    def append(self, state, action, action_logprob, next_state, reward, done):
        self.transition_dict['states'].append(state)  # np.ndarray
        self.transition_dict['actions'].append(action)  # np.ndarray
        self.transition_dict['action_logprobs'].append(action_logprob)  # float
        self.transition_dict['next_states'].append(next_state)  # np.ndarray
        self.transition_dict['rewards'].append(reward)  # float
        self.transition_dict['dones'].append(done)  # Bool

    def collect(self):
        states = torch.tensor(np.stack(self.transition_dict['states']), dtype=torch.float32).to(device)
        actions = torch.tensor(np.stack(self.transition_dict['actions']), dtype=torch.float32).to(device)
        action_logprobs = torch.tensor(self.transition_dict['action_logprobs'], dtype=torch.float32).to(device)
        next_states = torch.tensor(np.stack(self.transition_dict['next_states']), dtype=torch.float32).to(device)
        rewards = self.transition_dict['rewards']
        dones = self.transition_dict['dones']
        return states, actions, action_logprobs, next_states, rewards, dones

    def clear(self):
        for key in self.transition_dict.keys():
            self.transition_dict[key] = []


class Actor(nn.Module):
    def __init__(self, state_dim_h, action_dim, hidden_dim, action_std_init=0.6):
        super(Actor, self).__init__()

        self.state_dim_h = state_dim_h
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_var = torch.full((self.action_dim,), action_std_init ** 2).to(device)

        # self.rnn_layers = 2
        self.rnn = CfC(state_dim_h, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        current_batch_size = state.shape[0]
        h_0 = torch.zeros(current_batch_size, self.hidden_dim).to(device)
        rnn_output, _ = self.rnn(state, h_0)  # rnn_output: (batch_size, time_seq, hidden_size)

        normal_action = self.fc(rnn_output[:, -1])
        return normal_action

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std ** 2).to(device)

    def act(self, state):
        action_mean = self.forward(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)  # 针对不带batch_size的tensor
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def select_action(self, state):  # state: without batch_size
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action_mean = self.forward(state).squeeze(0)
            cov_mat = torch.diag(self.action_var)  # 针对不带batch_size的tensor
            dist = MultivariateNormal(action_mean, cov_mat)

            action = dist.sample()
            action_logprob = dist.log_prob(action)  # 标量tensor

        return action.detach().cpu().numpy(), action_logprob.detach().item()

    def evaluate(self, state, action):
        action_mean = self.forward(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)  # 针对带有batch_size的tensor
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy


class Critic(nn.Module):
    def __init__(self, state_dim_h, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.state_dim_h = state_dim_h
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # self.rnn_layers = 2
        self.rnn = CfC(self.state_dim_h, self.hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, state):
        current_batch_size = state.shape[0]
        h_0 = torch.zeros(current_batch_size, self.hidden_dim).to(device)
        rnn_output, _ = self.rnn(state, h_0)  # rnn_output: (batch_size, time_seq, hidden_size)

        value = self.fc(rnn_output[:, -1])
        return value


class PPO:
    def __init__(self, state_dim_h, state_keep_n, action_dim, hidden_dim=256):
        self.action_std = ACTION_STD_INIT

        self.buffer = Buffer()

        self.actor = Actor(state_dim_h, action_dim, hidden_dim, action_std_init=self.action_std).to(device)
        self.critic = Critic(state_dim_h, action_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # self.actor_old = Actor(state_dim_h, action_dim, hidden_dim, action_std_init=self.action_std).to(device)
        # self.actor_old.load_state_dict(self.actor.state_dict())

        self.mseLoss = nn.MSELoss()

        self.checkpoint = {'actor_state_dict': self.actor.state_dict(),
                           # 'actor_old_state_dict': self.actor_old.state_dict(),
                           'critic_state_dict': self.critic.state_dict(),
                           'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                           'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                           }

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.actor.set_action_std(self.action_std)
        # self.actor_old.set_action_std(self.action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        self.action_std -= action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)
        print("--------------------------------------------------------------------------------------------")

    def update(self):
        old_states, old_actions, old_logprobs, next_states, rewards, dones = self.buffer.collect()

        # Monte Carlo estimate of returns
        discounted_rewards = [0] * len(rewards)  # Pre-allocate space for efficiency
        R = 0  # Initialize the post-state return
        for i in reversed(range(len(rewards))):
            R = rewards[i] + GAMMA * R * (1 - dones[i])  # Reset R at the end of each episode
            discounted_rewards[i] = R

        # Normalizing the rewards
        rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        with torch.no_grad():
            old_state_values = self.critic(old_states).squeeze(-1).detach()

        # calculate advantages
        advantages = rewards - old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(K_EPOCHS):
            # Evaluating old actions and values
            logprobs, dist_entropy = self.actor.evaluate(old_states, old_actions)  # new policy
            state_values = self.critic(old_states).squeeze(-1)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            # calculate loss
            actor_loss = torch.mean(-torch.min(surr1, surr2) - 0.01 * dist_entropy)
            critic_loss = 0.5 * self.mseLoss(state_values, rewards)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # Copy new weights into old policy
        # self.actor_old.load_state_dict(self.actor.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, model_save_path):
        torch.save(self.checkpoint, model_save_path)

    def load(self, model_save_path):
        model_data = torch.load(model_save_path, map_location=lambda storage, loc: storage)
        self.actor.load_state_dict(model_data['actor_state_dict'])
        # self.actor_old.load_state_dict(model_data['actor_old_state_dict'])
        self.critic.load_state_dict(model_data['critic_state_dict'])
        self.actor_optimizer.load_state_dict(model_data['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(model_data['critic_optimizer_state_dict'])


################################### Training ###################################
def train():
    """Initialization"""
    env_wrapper = DroneEnvWrapper(render=True)
    ppo = PPO(state_dim_h=STATE_DIM_H,
              state_keep_n=STATE_KEEP_N,
              action_dim=ACTION_DIM)

    """Checkpoint"""
    save_directory = "./saved_model/PPO_saving"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    model_save_path = os.path.join(save_directory, f'{NAME}_model.pth')
    if os.path.exists(model_save_path):
        ppo.load(model_save_path)
        print(f'Previous model loaded: {model_save_path}')

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    time_step = 0  # current time_step
    n_episode = 0  # current episode
    reward_per_ep = []  # 用于记录每个EP的reward，统计变化
    distance_reward_per_ep = []  # 用于记录每个EP的reward，统计变化
    detection_reward_per_ep = []  # 用于记录每个EP的reward，统计变化

    # training loop
    while time_step <= MAX_TRAINING_TIMESTEPS:
        t1 = time.time()
        episode_step = 0
        state, start_position = env_wrapper.reset()  # 初始化
        if state is None:
            continue

        for t in range(1, MAX_EP_LEN + 1):

            # select action with policy
            action, action_logprob = ppo.actor.select_action(state)
            next_state, reward, done, successful, position = env_wrapper.step(action)

            # 异常episode
            if t < 15 and done:
                break

            # saving reward and is_terminals
            ppo.buffer.append(state, action, action_logprob, next_state, reward, done)

            state = next_state
            time_step += 1
            episode_step += 1

            # update PPO agent
            if time_step % UPDATE_TIMESTEP == 0:
                ppo.update()

            # if continuous action space; then decay action std of ouput action distribution
            if time_step % ACTION_STD_DECAY_FREQ == 0:
                ppo.decay_action_std(ACTION_STD_DECAY_RATE, MIN_ACTION_STD)

            # save model weights
            if time_step % SAVE_MODEL_FREQ == 0:
                print("--------------------------------------------------------------------------------------------")
                ppo.save(model_save_path)
                print("Model saved at : " + model_save_path)
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        n_episode += 1
        reward_per_ep.append(env_wrapper.episode_reward)
        distance_reward_per_ep.append(env_wrapper.episode_distance_reward)
        detection_reward_per_ep.append(env_wrapper.episode_detection_reward)
        print('\rEpisode: {}/{} | '
              'Iteration: {} | '
              'Episode Reward: {:.4f} | '
              'Distance Reward: {:.4f} | '
              'Detection Reward: {:.4f} | '
              'Final Reward: {:.4f} | '
              'Step: {} | '
              'Running Time: {:.2f}'
              .format(n_episode, MAX_TRAINING_EPISODES,
                      time_step,
                      env_wrapper.episode_reward,
                      env_wrapper.episode_distance_reward,
                      env_wrapper.episode_detection_reward,
                      env_wrapper.episode_final_reward,
                      episode_step, time.time() - t1))

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
