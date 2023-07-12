import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer
from environemnt import Environment
from Q_table import func_action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Network, self).__init__()
        # n_observations --> 15
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        # n_actions --> 48
        self.layer4 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        return F.relu(self.layer4(x))


class Double_DQN:

    def __init__(self,
                 n_agents,
                 n_features,
                 n_actions,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy_start=0.9,
                 e_greedy_end=0.15,
                 replace_target_iter=300,
                 memory_size=1500,
                 batch_size=32,
                 e_greedy_decrease=0.001,
                 epoch=100):
        # print("fea:", n_features)
        self.n_agents = n_agents
        self.n_features = n_features
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy_start
        self.epsilon_end = e_greedy_end
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_decrease = e_greedy_decrease
        self.epoch = epoch

        self.learn_step_counter = 0

        # 初始化replay
        self.memory = ReplayBuffer(self.memory_size)

        self.cost_his = []

        self.eval_net = [None for _ in range(self.n_agents)]
        self.target_net = [None for _ in range(self.n_agents)]
        self.optimizer = [None for _ in range(self.n_agents)]

        for i in range(self.n_agents):

            self.eval_net[i], self.target_net[i] = Network(
                self.n_features,
                self.n_actions).to(device), Network(self.n_features,
                                                    self.n_actions).to(device)
            self.optimizer[i] = optim.Adam(self.eval_net[i].parameters(),
                                           lr=learning_rate)

        # self.loss_fun = nn.MSELoss()
        self.loss_fun = nn.SmoothL1Loss()

    def store_memory(self, s, a, r, s_):
        self.memory.add(s, a, r, s_)

    # 返回的是一个动作向量的代表

    def choose_action(self, observation):
        a = []
        for i in range(self.n_agents):
            # observation = np.array(observation[i]).reshape(1, self.n_features)
            # obs = np.array(observation[i]).reshape(1, self.n_features)
            # # 增加一个维度 i.e[1,2,3,4,5]变成[[1,2,3,4,5]]
            # obs = torch.FloatTensor(obs[:])
            obs = torch.tensor(observation[i],
                               device=device,
                               dtype=torch.float32).unsqueeze(0)
            self.epsilon = self.epsilon - \
                self.epsilon_decrease if self.epsilon > self.epsilon_end else self.epsilon_end
            # 贪婪算法，以1-ε的概率选择最大动作
            if np.random.uniform() > self.epsilon:
                with torch.no_grad():
                    # 选择q值最大的动作
                    actions_value = self.eval_net[i](obs)
                    action = torch.max(actions_value,
                                       1)[1].data.cpu().detach().numpy()
                    action = action[0]
            else:
                action = np.random.randint(0, self.n_actions)
            a.append(action)
        return a

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            for i in range(self.n_agents):
                self.target_net[i].load_state_dict(
                    self.eval_net[i].state_dict())  # 直接赋值更新权重
        self.learn_step_counter += 1

        for agent_idx, (agent_eval, agent_target, opt) in \
                enumerate(zip(self.eval_net, self.target_net, self.optimizer)):
            # 随机抽样
            state, action, reward, next_state = self.memory.sample(
                self.batch_size, agent_idx, option=False)
            # print(action)
            # print(action.shape)
            # actions_index = []
            # 这边是针对batch_size和n_agents的经验回放池
            # rew --> (batch_size,1)
            rew = torch.tensor(reward, device=device, dtype=torch.float)
            # print(rew.size())
            # rew = rew.reshape(self.batch_size, self.n_agents, 1)
            # act --> (batch_size, n_agents)
            act = torch.from_numpy(action).to(device, torch.int64)
            # print(act)
            act = act.reshape(self.batch_size, self.n_agents, 1)
            # print(act.size())
            # obs --> (batch_size, n_agents*n_features)
            obs_n = torch.from_numpy(state).to(device, torch.float)
            obs_n_ = torch.from_numpy(next_state).to(device, torch.float)
            obs_n = obs_n.reshape(self.batch_size, self.n_agents,
                                  self.n_features)
            # print(obs_n.size())
            # print(obs_n)
            obs_n_ = obs_n_.reshape(self.batch_size, self.n_agents,
                                    self.n_features)
            # q_eval --> (batch_size, n_agents, 1)
            q_eval = agent_eval(obs_n).gather(-1, act)
            q_eval = q_eval.reshape(-1)
            # print(q_eval.size())
            # print(q_eval)
            
            q_next = agent_target(obs_n_).detach()
            q_next = q_next.max(2)[0]

            # print(q_next.size())

            q_target = q_next * self.gamma + rew
            q_target = q_target.reshape(-1)
            loss = self.loss_fun(q_eval, q_target)
            # print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()


#
# n_agents = 15
# env = Environment(15, 4)
# DDQN = Double_DQN(15, 15, 48)
# t_action_dqns = [[] for _ in range(n_agents)]
# epochs = []
# epoch_times = []
# epoch_wrong_times = []
# epoch_utilization = []
# game_step = 0
# EPOCH = 500
# for epoch in range(EPOCH):
#     epochs.append(epoch)
#     states = env.reset()
#     done = False
#     time_step_average = []
#     utilization_step_average = []
#     success_times_step_average = []
#     print('第', epoch, '步开始训练')
#     while not done:
#         action_dqns = DDQN.choose_action(states)
#         for i in range(n_agents):
#             t_action_dqns[i] = func_action(action_dqns[i])
#         next_states, rewards, times, utilization_rates, wrong_times, done = env.step(
#             t_action_dqns)
#         DDQN.store_memory(states, action_dqns, rewards, next_states)
#         # print(action_dqns)
#         states = next_states
#         # time_step_average = np.mean(times)
#         wrong = np.array(wrong_times)
#         time_step_average.append(np.mean(times))
#         utilization_step_average.append(np.mean(utilization_rates))
#         success_times_step_average.append(np.mean(1-wrong))
#         if game_step >= 500:
#             DDQN.learn()
#         game_step += 1
#     epoch_times.append(np.mean(time_step_average))
#     epoch_wrong_times.append(np.mean(success_times_step_average))
#     epoch_utilization.append(np.mean(utilization_step_average))

# def smooth(data, weight=0.9):
#     '''用于平滑曲线，类似于Tensorboard中的smooth

#     Args:
#         data (List):输入数据
#         weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

#     Returns:
#         smoothed (List): 平滑后的数据
#     '''
#     last = data[0]  # First value in the plot (first timestep)
#     smoothed = list()
#     for point in data:
#         smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
#         smoothed.append(smoothed_val)
#         last = smoothed_val
#     return smoothed

# plt.figure()
# plt.title('delay')
# plt.xlabel('epoch')
# plt.ylabel('time')
# plt.plot(epochs, smooth(epoch_times), label='DDQN')
# plt.legend()

# plt.figure()
# plt.title('success')
# plt.xlabel('epoch')
# plt.ylabel('success')
# plt.plot(epochs, smooth(epoch_wrong_times), label='DDQN')
# plt.legend()
# plt.show()
