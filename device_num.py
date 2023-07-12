import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['font.family'] = 'SimSun'
# plt.rcParams['font.sans-serif'] = ['SimSum']
x = np.array([13, 15, 17, 19, 21, 23])
# ddpg_steps = np.array([400, 430, 450, 452, 447, 450])
# ddqn_steps = np.array([230, 300, 360, 410, 460, 500])
# duelingdqn_steps = np.array([250, 320, 370, 440, 480, 520])
# plt.figure()
# plt.xlabel('终端设备数量/个', fontproperties='SimSun', fontsize=15)
# plt.ylabel('收敛步数', fontproperties='SimSun', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', size=14)
# plt.xticks(fontproperties='Times New Roman', size=14)
# plt.xticks(range(11, 25, 2))
# plt.plot(x, ddqn_steps, marker='^', ms='8', label='DDQN')
# plt.plot(x, duelingdqn_steps, marker='s', ms='8',
#          color='black', label='Dueling DQN')
# plt.plot(x, ddpg_steps, marker='o', ms='8', color='red', label='DDPG')
# plt.legend(prop={'family': 'Times New Roman', 'size': 14})
# plt.grid()
# plt.show()
# dueling_steps = np.array([200,260,350,350,])
# ddpg_success = np.array([])

# 成功率
# ddpg_success = np.array([0.82, 0.85, 0.83, 0.86, 0.82, 0.8])
# d3qn_success = np.array([0.80, 0.75, 0.66, 0.64, 0.60, 0.60])
# ddqn_success = np.array([0.70, 0.62, 0.50, 0.45, 0.44, 0.45])
# plt.figure()
# plt.xlabel('终端设备数量/个', fontproperties='SimSun', fontsize=15)
# plt.ylabel('成功率', fontproperties='SimSun', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', size=14)
# plt.xticks(fontproperties='Times New Roman', size=14)
# plt.xticks(range(11, 25, 2))
# plt.plot(x, ddqn_success, marker='^', ms='8', label='DDQN')
# plt.plot(x, d3qn_success, marker='s', ms='8',
#          color='black', label='Dueling DQN')
# plt.plot(x, ddpg_success, marker='o', ms='8', color='red', label='DDPG')
# plt.legend(prop={'family': 'Times New Roman', 'size': 14})
# plt.grid()
# plt.show()

# 负载方差
ddpg_differ = np.array([0.055, 0.06, 0.062, 0.063, 0.065, 0.068])
dueling_differ = np.array([0.087, 0.10, 0.11, 0.114, 0.12, 0.125])
ddqn_differ = np.array([0.059, 0.07, 0.075, 0.076, 0.082, 0.090])
plt.figure()
plt.xlabel('终端设备数量/个', fontproperties='SimSun', fontsize=15)
plt.ylabel('服务器负载方差', fontproperties='SimSun', fontsize=15)
plt.yticks(fontproperties='Times New Roman', size=14)
plt.xticks(fontproperties='Times New Roman', size=14)
plt.xticks(range(11, 25, 2))
plt.plot(x, ddqn_differ, marker='^', ms='8', label='DDQN')
plt.plot(x, dueling_differ, marker='s', ms='8',
         color='black', label='Dueling DQN')
plt.plot(x, ddpg_differ, marker='o', ms='8', color='red', label='DDPG')
plt.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 14})
plt.grid()
plt.show()
