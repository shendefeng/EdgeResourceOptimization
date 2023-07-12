import numpy as np
import matplotlib.pyplot as plt
# 参数
Noise = -115  # 单位dBm/Hz
# Channel_Gain = 0.5
# 单位GHz
FMAX = 12
# 单位W
Trans_Power_max = 24
# Trans_Power = 0.01  # mW
# 单位G
storage_max = 2.4  # 单位M
# 使用OFDM平分带宽数量
BandWith = 10

# 移动设备位置
Servers_num = 4
Server_Location = np.array(([100, 100], [100, 300], [300, 100], [300, 300]))
# print(Server_Location[1])

Device_Location = np.array(
    ([10, 300], [50, 50], [50, 240], [70, 200], [80, 350], [150, 100], [200, 40], [170, 300], [200, 250],
     [150, 400], [330, 330], [160, 200], [160, 200], [240, 310], [200, 200], [200, 210], [300, 200], [
         240, 100], [250, 350], [230, 180], [350, 100],
     [350, 300], [370, 220], [260, 260]))
Device_Num = len(Device_Location)
print(Device_Num)
# Device_Num = 15
# Device_Location = np.random.randint(100, 600, [Device_Num, 2])

# 覆盖范围
server_radius = 200
# print(Server_Location)
# print(Device_Location)

# 绘图
plt.rcParams['font.sans-serif'] = ['SimHei']
fig = plt.figure(figsize=(12, 12))
ax = plt.subplot(1, 1, 1)
ax.set_title("移动边缘计算仿真场景", fontsize=25)
ax.set_xlim((-100, 500))
ax.set_ylim((-100, 500))
ax.set_xlabel("x/m", fontsize=20)
ax.set_ylabel("y/m", fontsize=20)
ax.scatter(Server_Location[:, 0], Server_Location[:, 1], color='red',
           marker='^', s=80, label='边缘服务器')
X = Server_Location[:, 1]
Y = Server_Location[:, 1]
X = np.array([100, 100, 300, 300])
Y = np.array([100, 300, 100, 300])
for i, (x, y) in enumerate(zip(X, Y)):
    circle = plt.Circle((x, y),
                        server_radius, color='black', fill=False, linestyle='--')
    if i == 0:
        circle.set_label('边缘服务器覆盖范围')
    plt.gcf().gca().add_artist(circle)
ax.scatter(Device_Location[:, 0], Device_Location[:, 1], color='blue',
           marker='o', s=50, label='终端设备')
plt.rcParams.update({'font.size': 20})
ax.legend()
# ax.add_artist(leg)
plt.show()

# Device_Location = np.zeros((Device_Num, 2))
# for i in range(Device_Num):
#     Device_Location[i] = np.array(
#         ([np.random.randint(0, 500), np.random.randint(0, 500)]))
