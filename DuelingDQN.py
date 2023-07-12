import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        # V值
        self.v = nn.Linear(64, 1)
        # A值
        self.layer4 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        V = self.v(x)
        V = F.relu(V)
        A = self.layer4(x)
        A = F.relu(A)
        Q = V + A - torch.mean(A, axis=-1, keepdim=True)
        # print(Q.size())
        Q = F.relu(Q)
        return Q


class D3QN:

    def __init__(self,
                 n_agents,
                 n_features,
                 n_actions,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy_start=0.9,
                 e_greedy_end=0.1,
                 replace_target_iter=300,
                 memory_size=1500,
                 batch_size=64,
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
# write = SummaryWriter(log_dir="logs")
# #
# env = ENV(3, 3, 11, 1)
# # DQN = Double_DQN(env, env.n_actions, env.n_features*5)
# DQN = Double_DQN(env)
# epoch_reward = [0.0]
# epoch_average_reward = []
# for epoch in range(1000):
#     observation = env.reset()
#     epoch_average_reward.append(epoch_reward[-1]/ (env.UEs * 100))
#     epoch_reward.append(0)
#     print("epoch:{}, cost:{}".format(epoch, epoch_average_reward[epoch]))
#     # print("reset")
#     for step in range(100):
#         o1 = copy.deepcopy(observation)
#         o2 = copy.deepcopy(observation)
#
#         action = DQN.choose_action(o1)
#         o_, reward = env.step(o2, action, is_prob=False, is_compared=False)
#         epoch_reward[-1] += np.sum(reward)
#         DQN.store_memory(o2, action, reward, o_)
#         DQN.learn(epoch, write)
#         observation = o_
#     # print("action:", action)
