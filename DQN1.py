import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import copy
import os
from replay_buffer import *
from replay import ReplayBuffer
from environment2 import ENV
from network import OUActionNoise
from DQN_table2 import func_action
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
3层卷积神经网络
'''

# 可以有name的属性和标签


class Network(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer1.weight.data.normal_(0, 0.3)
        self.layer1.bias.data.normal_(0.1)
        self.bn1 = nn.LayerNorm(64)
        self.layer2 = nn.Linear(64, 128)
        self.layer2.weight.data.normal_(0, 0.3)
        self.layer2.bias.data.normal_(0.1)
        self.bn2 = nn.LayerNorm(128)
        self.layer3 = nn.Linear(128, n_actions)
        self.layer3.weight.data.normal_(0, 0.3)
        self.layer3.bias.data.normal_(0.1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return self.layer3(x)


class DDQN:
    def __init__(self, env, n_features, n_actions):
        self.batch_size = 64
        self.memory_size = 500
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.tau = 0.9
        self.lr = 0.01
        self.env = env
        # 输入，输出
        self.n_features = n_features
        self.n_actions = n_actions
        self.policy_net = Network(self.n_features, self.n_actions).to(device)
        self.target_net = Network(self.n_features, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayBuffer(self.memory_size, self.n_features, 1)
        self.criterion = nn.MSELoss()
        self.steps_done = 0
        self.learn_step_counter = 0
        self.q_net_iteration = 50
        self.noise = OUActionNoise(mu=np.zeros(self.n_actions))

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32,
                             device=device).unsqueeze(0)
        sample = random.random()
        # self.eps_threshold = self.eps_end + \
        #     (self.eps_start - self.eps_end) * \
        #     math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > self.eps_start:
            with torch.no_grad():
                action = self.policy_net(state)
                action = torch.max(action, 1)[1].data.cpu().detach().numpy()
                action = action[0]
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, next_state, done = \
            self.memory.sample_buffer(self.batch_size)
        reward_batch = torch.tensor(reward, device=device, dtype=torch.float)
        done = torch.tensor(done)
        next_state_bacth = torch.tensor(
            next_state, device=device, dtype=torch.float)
        action_batch = torch.tensor(action, device=device, dtype=torch.int64)
        # print(action)
        state_batch = torch.tensor(state, device=device, dtype=torch.float)
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_bacth).detach()
        next_state_values = next_state_values.max(
            1)[0].view(self.batch_size, 1)
        expected_state_action_values = (
            next_state_values * self.gamma) + reward_batch.unsqueeze(1)
        loss = self.criterion(state_action_values,
                              expected_state_action_values)
        opt = self.optimizer
        opt.zero_grad()
        loss.backward()
        # 这一段代码是什么情况
        # 裁剪梯度，阈值设置为100
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        opt.step()
        # 每隔10步更新网络
        # if self.learn_step_counter % self.q_net_iteration == 0:
        self.update_network_parameters()
        # self.learn_step_counter += 1

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def update_network_parameters(self):
        TAU = self.tau
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)


num_task = 15
env = ENV(15,  15, 4)
# # # DQN = Double_DQN(env, env.n_actions, env.n_features*5)
agent = DDQN(env, 87, 48)
score_record = []
score = 0
score_record_step = []
time_record = []
load_record = []
for epoch in range(500):
    done = False
    state = env.reset()
    score = 0
    times = 0
    load = 0
    while not done:
        action = agent.select_action(state)
        t_action = func_action(action)
        next_state, reward, done, time, utilization = env.step(t_action)
        score += reward
        times += time
        load += utilization
        agent.store_transition(state, action, reward, next_state, int(done))
        state = next_state

        agent.learn()
#             # 网络参数软更新
    score_record.append(score)
    time_record.append(times/num_task)
    load_record.append(load/num_task)
    print('epoch ', epoch, '\tscore %.2f' %
          score, "\twrong: ", env.count_wrong)
    print('episode ', epoch, 'delay%.2f' %
          times, 'load%.2f' % load)
    if epoch % 25 == 0:
        score_record_step.append(np.mean(score_record))
# reward


def smooth(data, weight=0.9):
    '''用于平滑曲线，类似于Tensorboard中的smooth

    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


plt.figure()
plt.title('Reward')
x_data = range(len(score_record))
plt.xlabel('episodes')
plt.ylabel('reward')
plt.plot(x_data, score_record)
plt.plot(x_data, smooth(score_record))
plt.figure()
plt.title('delay')
x_data = range(len(time_record))
plt.xlabel('episodes')
plt.ylabel('delay')
plt.plot(x_data, time_record, label='delay')
plt.plot(x_data, smooth(time_record), label='smooth-dalay')

plt.show()
# plt.figure()
# plt.title('Reward Steps')
# x_data = range(len(score_record_step))
# plt.xlabel('episodes steps')
# plt.ylabel('reward steps')
# plt.plot(x_data, score_record_step)
# plt.show()
#             epoch_reward.append(reward_sum)
#     print('epoch:', epoch, '\treward:', reward_sum)
# print('epoch:', epoch, '\taction:', action, '\treward:', reward)
