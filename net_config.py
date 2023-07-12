"""
移动设备，(任务)，基站
"""

import random
import numpy as np
from constant import *

# 任务：数据量Mbit，cpu工作量MCycle，时延约束(0.50s)


class Task:
    def __init__(self):
        self.data_size = round(random.uniform(6, 8), 2)
        self.cpu_cycles = round(random.uniform(2, 5), 2)
        self.delay_constraints = 0.70
# 移动设备：计算能力MHz，位置


def get_device_info(num):
    devices = {}
    i = 0
    while i < num:
        task = Task()
        devices[i] = Device(task, Device_Location[i])
        i += 1
    return devices


class Device:
    def __init__(self, task, location):
        self.task = task
        self.device_location = location

    # def Set_Location()
# device = Device()


# devices = get_device_info(10)
# for i in range(4):
#     print(devices[i].task.data_size)
#     print(devices[i].task.cpu_cycles)
#     print(devices[i].task.delay_constraints)
#     print(devices[i].MD_location)
# 任务：数据量Mbit，cpu工作量MCycle，时延约束(0.50s)


# task = Task()
# 基站：计算能力10GHz,传输功率(11W),带宽(RB资源块),存储空间(10G),位置,覆盖范围
# 需要补充初始化情况


def get_mec_info(num):
    mecs = {}
    i = 0
    while i < num:
        mecs[i] = MEC(FMAX, Trans_Power_max, BandWith,
                      storage_max, Server_Location[i])
        # bs[i] = BS(random.randint(6000, 8000),
        #            random.uniform(0.002, 0.003), random.uniform(1, 2), Server_Location[i])
        i += 1
    return mecs


class MEC:
    def __init__(self, cpu_frequency, trans_power, bandwith, storage, locations):
        self.cpu_frequency = cpu_frequency
        self.trans_power = trans_power
        self.bandwith = bandwith
        self.storage = storage
        self.Servers_Location = locations
        self.MEC_Radius = 200


# 计算距离
# server = get_mec_info(2)
# device = get_device_info(2)
# print(server[0].Servers_Location)
# print(device[0].device_location)
# distance = np.sqrt((server[0].Servers_Location[0]-device[0].device_location[0])
#                    ** 2+(server[0].Servers_Location[1]-device[0].device_location[1])**2)
# print(distance)

# 测试
# servers = get_bs_info(4)
# for i in range(2):
#     print(servers[i].cpu_frequency)
#     print(servers[i].Servers_Location)
