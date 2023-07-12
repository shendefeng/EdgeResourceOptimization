import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

import matplotlib as mpl

# mpl.use('Agg')

# font = font_manager.FontProperties(fname="/usr/share/fonts/simhei.ttf")
# a = np.array([2, 3, 4, 5])
# b = np.sum(a)/len(a)
# print(b)
# a = [[] for _ in range(15)]
# j = 0
# for i in range(15):
#     a[i].append(1)

# for i in range(15):
#     a[i].append(2)

# # b = np.array(a)
# print(a)
# print(b)
# a2 = [0.1, 0.2, 0.3, 0.4, 0.5]
# a = np.array(a2)
# b = np.mean(1-a)
# print(b)
# obs = [4 for _ in range(15)]
# act = [1 for _ in range(15)]
# print(len(obs))
# # print(len(obs[0]))
# obs_size = []
# action_size = []
# head_o, head_a, end_o, end_a = 0, 0, 0, 0
# for obs_i, act_i in zip(obs, act):
#     end_o = end_o + obs_i
#     end_a = end_a + act_i
#     range_o = (head_o, end_o)
#     range_a = (head_a, end_a)
#     obs_size.append(range_o)
#     action_size.append(range_a)
#     head_o = end_o
#     head_a = end_a
# print(len(obs_size))
# print(len(obs_size[0]))
# print(obs_size)

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.figure()
# plt.xlabel('我/m', fontproperties=font, fontsize=15)
# plt.ylabel('你/m', fontproperties=font, fontsize=15)
# a = [12, 3, 90, 10]
# b = [1, 2, 3, 4]
# c = [10, 10, 10]
# plt.plot(b, a)
# # plt.show()
# plt.savefig('table1.png')

# plt.figure()
# plt.plot(b, c)
# plt.savefig('table2.png')
# plt.show()
# a = np.array([0.8, 0.4, 0.1, 0.4])
# b = 2 * np.mean(a)
# print(b)

# a = np.array([[[0.8], [0.2]], [[0.3], [0.4]], [[0.2], [0.5]]])
# print(a)


epoch = 100
a = []
b = []
c = []
d = []
for i in range(epoch):
    a.append(i)
    x = i ^ 2-2*i
    y = x - 30
    z = x + 30
    b.append(x)
    c.append(y)
    d.append(z)

plt.figure()
plt.plot(a, b, linewidth=2.0,  linestyle='--', label='DDQN')
plt.plot(a, c, linewidth=2.0,  linestyle='dotted',
         color='green', label='Dueling DQN')
plt.plot(a, d, linewidth=2.0,  color='red', label='DDPG')
plt.legend()
plt.grid()
plt.show()
