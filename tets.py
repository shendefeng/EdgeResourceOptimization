# import matplotlib as mpl

# mpl.use('Agg')
# from matplotlib import font_manager

# font = font_manager.FontProperties(fname="/usr/share/fonts/simhei.ttf")
import copy
from environemnt import Environment
from MADDPG import Maddpg
from replay_buffer import ReplayBuffer
from noises import OUNoise
from DDQN import Double_DQN
from DuelingDQN import D3QN
from Q_table import func_action
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 400
STEP = 200
batch_size = 64
memory_size = 1500
env = Environment(15, 4)
n_agents = 15
maddpg = Maddpg()
DDQN = Double_DQN(15, 15, 48)
DuelingDQN = D3QN(15, 15, 48)
noise = OUNoise(4)
print('=============================')
print('=1 Env {} is right ...')
print('=============================')

obs_shape_n = [env.n_features for _ in range(env.device_num)]
action_shape_n = [env.n_actions for _ in range(env.device_num)]
actors_cur, critic_cur, actors_tar, critic_tar, optimizers_a, optimizers_c = \
    maddpg.get_train(n_agents, obs_shape_n, action_shape_n)
memory_dpg = ReplayBuffer(memory_size)
game_step = 0
update_cnt = 0
obs_size = []
action_size = []
agent_rewards = [[] for _ in range(n_agents)]
t_action_dqns = [[] for _ in range(n_agents)]
t_action_d3qn = [[] for _ in range(n_agents)]
epochs = []
head_o, head_a, end_o, end_a = 0, 0, 0, 0
for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
    end_o = end_o + obs_shape
    end_a = end_a + action_shape
    range_o = (head_o, end_o)
    range_a = (head_a, end_a)
    obs_size.append(range_o)
    action_size.append(range_a)
    head_o = end_o
    head_a = end_a
epoch_rewards_MADDPG = []
epoch_times_MADDPG = []
epoch_utilization_MADDPG = []
epoch_differ_MADDPG = []
epoch_wrong_times_MADDPG = []
epoch_rewards_DDQN = []
epoch_times_DDQN = []
epoch_utilization_DDQN = []
epoch_differ_DDQN = []
epoch_wrong_times_DDQN = []
epoch_rewards_D3QN = []
epoch_times_D3QN = []
epoch_utilization_D3QN = []
epoch_differ_D3QN = []
epoch_wrong_times_D3QN = []
i = 0
for epoch in range(EPOCH):
    epochs.append(epoch)
    states = env.reset()
    states_maddpg = copy.deepcopy(states)
    states_dqn = copy.deepcopy(states)
    states_d3qn = copy.deepcopy(states)
    # done = False
    done_dqn = False
    rewards_step_average_MADDPG = []
    time_step_average_MADDPG = []
    utilization_step_average_MADDPG = []
    differ_step_MADDPG = []
    success_times_step_average_MADDPG = []
    rewards_step_average_DDQN = []
    time_step_average_DDQN = []
    utilization_step_average_DDQN = []
    differ_step_DDQN = []
    success_times_step_average_DDQN = []
    rewards_step_average_D3QN = []
    time_step_average_D3QN = []
    utilization_step_average_D3QN = []
    differ_step_D3QN = []
    success_times_step_average_D3QN = []
    print('epoch:\t', epoch)
    # while not done_dqn:
    for count in range(10):
        i += 1
        action_dqns = DDQN.choose_action(states_dqn)
        action_d3qn = DuelingDQN.choose_action(states_d3qn)
        action = [
            (agent(torch.from_numpy(observation).to(device, torch.float)) +
             torch.tensor(noise()).to(device,
                                      torch.float)).detach().cpu().numpy()
            for agent, observation in zip(actors_cur, states_maddpg)
        ]
        for i in range(n_agents):
            t_action_dqns[i] = func_action(action_dqns[i])
            t_action_d3qn[i] = func_action(action_d3qn[i])
        next_states_dqn, rewards_dqn, times_dqn, utilization_rates_dqn, util_differ_dqn, wrong_times_dqn, done_dqn = env.step(
            t_action_dqns)
        next_states_maddpg, rewards, times, utilization_rates, util_differ_MADDPG, wrong_times, done = env.step(
            action)
        next_states_d3qn, rewards_d3qn, times_d3qn, utilization_rates_d3qn, util_differ_d3qn, wrong_times_d3qn, done_d3qn = env.step(
            t_action_d3qn)
        memory_dpg.add(states_maddpg, np.concatenate(action), rewards,
                       next_states_maddpg)
        DDQN.store_memory(states_dqn, action_dqns, rewards_dqn,
                          next_states_dqn)
        DuelingDQN.store_memory(states_d3qn, action_d3qn, rewards_d3qn,
                                next_states_d3qn)
        states_maddpg = next_states_maddpg
        states_dqn = next_states_dqn
        states_d3qn = next_states_d3qn
        # 处理错误率
        wrong = np.array(wrong_times)
        wrong_dqn = np.array(wrong_times_dqn)
        wrong_d3qn = np.array(wrong_times_d3qn)
        # 处理reward
        # tmp1 = sorted(rewards, reverse=True)
        # six1 = tmp1[:10]
        # tmp_DDPG = sum(six1) / 10
        # tmp2 = sorted(rewards_dqn, reverse=True)
        # six2 = tmp2[:10]
        # tmp_DDQN = sum(six2) / 10
        # tmp3 = sorted(rewards_d3qn, reverse=True)
        # four3 = tmp3[:5]
        # tmp_D3QN = sum(four3) / 5
        # print('--------------------------------')
        # print(six1)
        # print(rewards_dqn)
        # print(rewards_d3qn)
        # MADDPG存储
        rewards_step_average_MADDPG.append(rewards)
        time_step_average_MADDPG.append(np.mean(times))
        utilization_step_average_MADDPG.append(np.mean(utilization_rates))
        success_times_step_average_MADDPG.append(np.mean(1 - wrong))
        differ_step_MADDPG.append(util_differ_MADDPG)
        # DDQN存储
        rewards_step_average_DDQN.append(rewards_dqn)
        time_step_average_DDQN.append(np.mean(times_dqn))
        utilization_step_average_DDQN.append(np.mean(utilization_rates_dqn))
        success_times_step_average_DDQN.append(np.mean(1 - wrong_dqn))
        differ_step_DDQN.append(util_differ_dqn)
        # D3QN存储
        rewards_step_average_D3QN.append(rewards_d3qn)
        time_step_average_D3QN.append(np.mean(times_d3qn))
        utilization_step_average_D3QN.append(np.mean(utilization_rates_d3qn))
        success_times_step_average_D3QN.append(np.mean(1 - wrong_d3qn))
        differ_step_D3QN.append(util_differ_d3qn)
        # 训练
        update_cnt, actors_cur, actors_tar, critic_cur, critic_tar = maddpg.agents_train(
            game_step, update_cnt, memory_dpg, obs_size, action_size,
            actors_cur, actors_tar, critic_cur, critic_tar, optimizers_a,
            optimizers_c)
        if game_step >= 600:
            DDQN.learn()
            DuelingDQN.learn()
        game_step += 1
        # print(game_step)
        # if i % 30 == 0:
        # step_times = np.mean(time_step_average)

        # print('episode\t', epoch, '\treward\t', rewards_step_average)
        # epoch_rewards.append(np.mean(rewards_step_average))
    # print(time_step_average)
    # print(rewards_step_average_MADDPG)
    # print(np.mean(rewards_step_average_MADDPG)*10)
    # print(rewards_step_average_DDQN)
    # print(np.mean(rewards_step_average_DDQN)*10)
    # print(rewards_step_average_D3QN)
    # print(np.mean(rewards_step_average_D3QN)*10)
    # MADDPG绘图数据
    epoch_rewards_MADDPG.append(np.mean(rewards_step_average_MADDPG) * 15)
    epoch_times_MADDPG.append(np.mean(time_step_average_MADDPG))
    epoch_wrong_times_MADDPG.append(np.mean(success_times_step_average_MADDPG))
    epoch_utilization_MADDPG.append(np.mean(utilization_step_average_MADDPG))
    epoch_differ_MADDPG.append(np.mean(differ_step_MADDPG))
    # DDQN绘图数据
    epoch_rewards_DDQN.append(np.mean(rewards_step_average_DDQN) * 15)
    epoch_times_DDQN.append(np.mean(time_step_average_DDQN))
    epoch_wrong_times_DDQN.append(np.mean(success_times_step_average_DDQN))
    epoch_utilization_DDQN.append(np.mean(utilization_step_average_DDQN))
    epoch_differ_DDQN.append(np.mean(differ_step_DDQN))
    # D3QN绘图数据
    epoch_rewards_D3QN.append(np.mean(rewards_step_average_D3QN) * 15)
    epoch_times_D3QN.append(np.mean(time_step_average_D3QN))
    epoch_wrong_times_D3QN.append(np.mean(success_times_step_average_D3QN))
    epoch_utilization_D3QN.append(np.mean(utilization_step_average_D3QN))
    epoch_differ_D3QN.append(np.mean(differ_step_D3QN))


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


# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure()
plt.xlabel('步数/步', fontproperties='SimSun', fontsize=15)
plt.ylabel('奖励', fontproperties='SimSun', fontsize=15)
# plt.xlabel('步数/步', fontsize=15)
# plt.ylabel('平均奖励', fontsize=15)
plt.plot(epochs, smooth(epoch_rewards_DDQN),
         linewidth=2.0,  linestyle='--', label='DDQN')
plt.plot(epochs, smooth(epoch_rewards_D3QN), linewidth=2.0,
         linestyle='dotted', color='green', label='Dueling DQN')
plt.plot(epochs, smooth(epoch_rewards_MADDPG),
         linewidth=2.0, color='red', label='DDPG')
# plt.rcParams.update({'fontsize': 20})
plt.legend(prop={'family': 'Times New Roman', 'size': 12})
plt.grid()
# plt.savefig('rewards-15.png')

plt.figure()
plt.xlabel('步数/步', fontproperties='SimSun', fontsize=15)
plt.ylabel('平均时延/s', fontproperties='SimSun', fontsize=15)
# plt.xlabel('步数/步', fontsize=15)
# plt.ylabel('平均时延/s', fontsize=15)
plt.plot(epochs, smooth(epoch_times_DDQN),
         linewidth=2.0,  linestyle='--', label='DDQN')
plt.plot(epochs, smooth(epoch_times_D3QN), linewidth=2.0,
         linestyle='dotted', color='green', label='Dueling DQN')
plt.plot(epochs, smooth(epoch_times_MADDPG),
         linewidth=2.0, color='red', label='DDPG')
# plt.rcParams.update({'fontsize': 20})
plt.legend(prop={'family': 'Times New Roman', 'size': 12})
plt.grid()
# plt.savefig('delay-15.png')

plt.figure()
plt.xlabel('步数/步', fontproperties='SimSun', fontsize=15)
plt.ylabel('卸载成功率', fontproperties='SimSun', fontsize=15)
# plt.xlabel('步数/步', fontsize=20)
# plt.ylabel('成功率', fontsize=20)
plt.plot(epochs, smooth(epoch_wrong_times_DDQN),
         linewidth=2.0,  linestyle='--', label='DDQN')
plt.plot(epochs, smooth(epoch_wrong_times_D3QN), linewidth=2.0,
         linestyle='dotted', color='green', label='Dueling DQN')
plt.plot(epochs, smooth(epoch_wrong_times_MADDPG),
         linewidth=2.0, color='red', label='DDPG')
# plt.rcParams.update({'fontsize': 20})
plt.legend(prop={'family': 'Times New Roman', 'size': 12})
plt.grid()
# plt.savefig('success-15.png')

plt.figure()
plt.xlabel('步数/步', fontproperties='SimSun', fontsize=15)
plt.ylabel('服务器平均负载', fontproperties='SimSun', fontsize=15)
# plt.xlabel('步数/步', fontsize=20)
# plt.ylabel('服务器平均负载', fontsize=20)
plt.plot(epochs, smooth(epoch_utilization_DDQN),
         linewidth=2.0,  linestyle='--', label='DDQN')
plt.plot(epochs, smooth(epoch_utilization_D3QN), linewidth=2.0,
         linestyle='dotted', color='green', label='Dueling DQN')
plt.plot(epochs, smooth(epoch_utilization_MADDPG),
         linewidth=2.0, color='red', label='DDPG')
# plt.rcParams.update({'fontsize': 20})
plt.legend(prop={'family': 'Times New Roman', 'size': 12})
plt.grid()
# plt.savefig('utilization_15.png')

plt.figure()
plt.xlabel('步数/步', fontproperties='SimSun', fontsize=15)
plt.ylabel('服务器负载方差', fontproperties='SimSun', fontsize=15)
# plt.xlabel('步数/步', fontsize=20)
# plt.ylabel('服务器平均负载', fontsize=20)
plt.plot(epochs, smooth(epoch_differ_DDQN),
         linewidth=2.0,  linestyle='--', label='DDQN')
plt.plot(epochs, smooth(epoch_differ_D3QN), linewidth=2.0,
         linestyle='dotted', color='green', label='Dueling DQN')
plt.plot(epochs, smooth(epoch_differ_MADDPG),
         linewidth=2.0, color='red', label='DDPG')
# plt.rcParams.update({'fontsize': 20})
plt.legend(prop={'family': 'Times New Roman', 'size': 12})
plt.grid()
# plt.savefig('differ_15.png')

plt.show()
