from net_config import *
from constant import *
'''
状态:device_num维度，每个维度6个[task.data_size,task.cpu_cycles,is_off, rest_resource,rest_trans,rest_storage]
动作:device_num维度，每个维度4个[is_off, alloc_resource,alloc_trans,alloc_storage]
奖励:device_num维度,reward_t
'''


class Environment:

    def __init__(self, device_num, server_num):
        self.device_num = device_num
        self.server_num = server_num
        self.devices = get_device_info(device_num)
        self.servers = get_mec_info(server_num)
        self.task_num = 10
        self.task_count = 1
        self.done = False
        # 单个智能体的feature和action,共有device_num个智能体
        self.n_features = 3 + 3 * server_num
        self.n_actions = 4
        # 单个服务器的最大资源
        self.fmax = FMAX
        self.pman = Trans_Power_max
        self.storagemax = storage_max
        self.bandwith = BandWith
        self.N0 = Noise  # 单位dBm
        self.delay_constrains = 0.70
        # 针对业务分配的资源 --> 动作
        self.is_off = [0] * self.device_num
        self.alloc_resource = [0] * self.device_num
        self.alloc_trans = [0] * self.device_num
        self.alloc_storage = [0] * self.device_num
        for i in range(self.device_num):
            self.is_off[i] = random.randint(0, self.server_num - 1)
            self.alloc_resource[i] = 0
            self.alloc_trans[i] = 0
            self.alloc_storage[i] = 0
        # 初始利用率
        self.com_util = [0] * server_num
        self.sto_util = [0] * server_num
        self.utilization_rate = [0] * server_num
        self.rest_resource = [0] * server_num
        self.rest_trans = [0] * server_num
        self.rest_storage = [0] * server_num
        for i in range(server_num):
            self.rest_resource[i] = self.servers[i].cpu_frequency
            self.rest_trans[i] = self.servers[i].trans_power
            self.rest_storage[i] = self.servers[i].storage
        self.wrong_times = 0
        self.time = 0

    # def init_servers(self):
    #     for i in range(self.server_num):
    #         self.rest_resource[i] = self.servers[i].cpu_frequency
    #         self.rest_trans[i] = self.servers[i].trans_power
    #         self.rest_storage[i] = self.servers[i].storage

    def reset(self):
        # 初始利用率
        self.wrong_times = 0
        self.com_util = [0] * self.server_num
        self.sto_util = [0] * self.server_num
        self.utilization_rate = [0] * self.server_num
        # for i in range(self.server_num):
        #     self.rest_resource[i] = self.servers[i].cpu_frequency
        #     self.rest_trans[i] = self.servers[i].trans_power
        #     self.rest_storage[i] = self.servers[i].storage
        # 这里需要对每一个智能体进行观察
        # 每个智能体需要观察的状态有  --> 4个
        # (任务数据量，卸载目标，剩余计算资源，剩余传输资源，剩余存储资源))
        state = []
        for i in range(self.device_num):
            # 初始随机选择一个目标
            rest_resource = []
            rest_trans = []
            rest_storage = []
            self.is_off[i] = random.randint(0, self.server_num - 1)
            # 服务器剩余资源
            for j in range(self.server_num):
                rest_resource.append(self.rest_resource[j])
                rest_trans.append(self.rest_trans[j])
                rest_storage.append(self.rest_storage[j])
            task_data_size = self.devices[i].task.data_size
            task_cpu = self.devices[i].task.cpu_cycles
            observation = np.array([task_data_size, task_cpu, self.is_off[i]])
            observation = np.hstack(
                (observation, rest_resource, rest_trans, rest_storage))
            state.append(observation)
        state = np.array(state)
        return state

    # 测试任务
    # def get_action_test(self):
    #     agents_action = []
    #     action = [0] * self.n_actions

    #     for i in range(self.device_num):
    #         for j in range(4):
    #             action[j] = round(random.uniform(0, 1), 2)
    #         agents_action.append(action)
    #     return agents_action

    def new_task(self):
        for i in range(self.device_num):
            self.devices[i].task.data_size = round(random.uniform(6, 8), 2)
            self.devices[i].cpu_cycles = round(random.uniform(2, 5), 2)

    def reset_server(self):
        for i in range(self.server_num):
            self.rest_resource[i] = self.servers[i].cpu_frequency
            self.rest_trans[i] = self.servers[i].trans_power
            self.rest_storage[i] = self.servers[i].storage

    def step(self, actions):
        self.reset_server()
        # 定义时延，负载
        T = self.delay_constrains
        time = [0] * self.device_num
        com_util = [0] * self.server_num
        sto_util = [0] * self.server_num
        utilization_rate = [0] * self.server_num
        off_num = [0] * self.server_num
        off_rate = [0] * self.server_num
        # 定义奖励
        reward_t = [0] * self.device_num
        reward_load = [0] * self.device_num
        reward = [0] * self.device_num
        # 定义分配错误
        count_wrong = [0] * self.device_num
        load_wrong = [0] * self.device_num

        state = []
        # 对每一个智能体的选择的动作进行交互作用
        for i in range(self.device_num):
            # 提取动作
            # 4个动作
            # (卸载目标，分配的计算资源，分配的功率，分配的空间)
            action = actions[i]
            # print(action)
            # j = 0
            # while j < self.n_actions:
            #     print(j)
            #     # print(action[j])
            #     # 限定范围，可以省略
            for j in range(self.n_actions):
                if action[j] > 1:
                    action[j] = 1
                if action[j] < 0:
                    action[j] = 0
            #     j += 1
            # get1，get2，get3，get4
            # print('第', i, '个移动设备(智能体):',
            #       action[0], action[1], action[2], action[3])
            if action[0] <= 0.25:
                get1 = int(0)
            elif 0.25 < action[0] <= 0.50:
                get1 = int(1)
            elif 0.50 < action[0] <= 0.75:
                get1 = int(2)
            else:
                get1 = int(3)
            # 计算1-2GHz，单位GHz
            get2 = (action[1] + 1)
            get2 = round(get2, 2)
            # 传输功率2 -- 4W，单位是W
            get3 = (action[2] * 2 + 2)
            get3 = round(get3, 2)
            # 存储空间200-400M,单位是G
            get4 = (action[3]) * 0.2 + 0.2
            get4 = round(get4, 2)
            # print('get1234', get1, get2, get3, get4)
            # 时延模型
            distance = np.sqrt((self.devices[i].device_location[0] -
                                self.servers[get1].Servers_Location[0])**2 +
                               (self.devices[i].device_location[1] -
                                self.servers[get1].Servers_Location[1])**2)
            # print('距离：\t', round(distance, 2))
            self.is_off[i] = get1
            off_num[get1] += 1
            # 不在覆盖范围里, --> 不分配资源，时延最大,这是损耗会最大
            if distance > self.servers[get1].MEC_Radius:
                reward_t[i] = 0
                count_wrong[i] = 1
                time[i] = T
                utilization_rate[get1] = 0
            else:
                # 分配的资源
                f = get2
                # 分配的传输功率--> 转化为dBm
                p1 = get3
                p = 10 * np.log10(get3 * 10**(3))
                # 计算路径损耗
                if distance <= 80:
                    distance = 80
                elif distance >= 150:
                    distance = 150
                # 这里工作频率约在2GHz
                # 损耗约在70dB左右之间,这里有点不好说？
                h = 98.45 + 27.25 * np.log10(distance * 10**(-3))
                B = self.bandwith
                # 噪声功率,-45dBm
                n0 = 10**(self.N0 / 10) * B * 10**(6)
                n0 = 10 * np.log10(n0)
                # 分配1-2个以内的资源块(一块资源1MHz)
                # 单位dB
                SNR = p - h - n0
                SNR = 10**(SNR / 10)
                rate = B * np.log2(1 + SNR)
                # print('速率是\t', rate)
                # 传输速率,这里需要调整 ,SNR尽量靠近10-15dB
                # 计算 + 传输
                t1 = self.devices[i].task.cpu_cycles / (f * 10)
                t2 = self.devices[i].task.data_size / rate
                t = t1 + t2
                # print('计算时延是:\t', t1, '传输时延是:\t', t2, '总时延是:\t', t)
                # 存储分配
                store = get4
                # 满足约束
                if f <= self.rest_resource[get1] and p1 <= self.rest_trans[
                        get1] and store <= self.rest_storage[get1] and t <= T:
                    self.alloc_resource[i] = f
                    self.rest_resource[get1] -= f
                    self.alloc_trans[i] = p1
                    self.rest_trans[get1] -= p1
                    self.alloc_storage[i] = store
                    self.rest_storage[get1] -= store
                    # 保证一个数量级，时延越小越好
                    reward_t[i] = round(3 * (self.delay_constrains - t), 2)
                    # print('======', i, '=======')
                    # print(reward_t[i])
                    time[i] = round(t, 2)
                    # print(time[i])
                else:
                    reward_t[i] = 0
                    count_wrong[i] = 1
                    time[i] = T
                    if f < self.rest_resource[get1]:
                        self.rest_resource[get1] -= f
                    else:
                        load_wrong[i] = 1
                        self.rest_resource[get1] = 0
                    if p1 < self.rest_trans[get1]:
                        self.rest_trans[get1] -= p1
                    else:
                        load_wrong[i] = 1
                        self.rest_trans[get1] = 0
                    if store < self.rest_resource[get1]:
                        self.rest_storage[get1] -= store
                    else:
                        load_wrong[i] = 1
                        self.rest_storage[get1] = 0
                # utilization_rate[get1] = 0.5*(1-self.rest_resource[get1]/self.fmax) + 0.5 * (
                #     1-self.rest_storage[get1]/self.storagemax)
        self.task_count += 1
        # 产生新的任务
        if self.task_count <= self.task_num:
            self.new_task()
            self.done = False
        else:
            self.done = True
            self.task_count = 1
        # 每个智能体观察状态

        state = []
        for i in range(self.device_num):
            # 初始随机选择一个目标
            rest_resource = []
            rest_trans = []
            rest_storage = []
            # 服务器剩余资源
            for j in range(self.server_num):
                rest_resource.append(self.rest_resource[j])
                rest_trans.append(self.rest_trans[j])
                rest_storage.append(self.rest_storage[j])
            task_data_size = self.devices[i].task.data_size
            task_cpu = self.devices[i].task.cpu_cycles
            observation = np.array([task_data_size, task_cpu, self.is_off[i]])
            observation = np.hstack(
                (observation, rest_resource, rest_trans, rest_storage))
            state.append(observation)
        state = np.array(state)
        # print('交互作用后的状态是\t', state)
        # 计算服务器的负载情况,使用方差表示
        for i in range(self.server_num):
            com_util[i] = 1 - self.rest_resource[i] / self.fmax
            sto_util[i] = 1 - self.rest_storage[i] / self.storagemax
            utilization_rate[i] = 0.5 * com_util[i] + 0.5 * sto_util[i]
        # 系统总平均负载
        util = np.array(utilization_rate)

        utilizations = np.var(util)
        # print('系统总平均负载:\t', utilizations)
        # 计算负载奖励,并和时延奖励加权
        for i in range(self.device_num):
            if utilizations <= 0.01:
                reward_load[i] = 1 - utilizations
            if 0.01 < utilizations < 0.10:
                reward_load[i] = 1 - 10 * utilizations
            else:
                reward_load[i] = -4 * utilizations
            # reward_load[i] = round(1 - utilizations, 2)
            # 分配错误对系统的整体负载是有影响的！
            # if count_wrong[i] == 1:
            #     reward_load[i] -= 0.1
            #     reward_t[i] -= 0.02
            # if load_wrong[i] == 1:
            #     reward_load[i] -= 0.05
            # print('时延奖励：\t', reward_t[i])
            # print('负载奖励：\t', reward_load[i])
            reward[i] = round(0.40 * reward_load[i] + 0.60 * reward_t[i], 2)
            if count_wrong[i] == 0:
                reward[i] += 0.25
            # print('加权奖励：\t', reward[i])
        # print('state\t', state)
        # print('reward\t', reward)
        # print('利用率\t', utilization_rate)
        # print('分配错误数量\t', count_wrong)
        # print('是否完成\t', self.done)
        return state, reward, time, utilization_rate, utilizations, count_wrong, self.done


# env = Environment(15, 4)
# obs_n = [env.n_features for i in range(env.device_num)]
# print(sum(obs_n))
# print(env.device_num)
# state = env.reset()
# print(sum(state))
# s = np.array([])
# for obs in state:
#     s = np.concatenate(s, obs)
# print(s)
# print(env.n_actions)
# action = env.get_action_test()
# print(env.device_num)
# for i in range(env.device_num):
#     print(action[i])
# print(len(state))
# for i in range(env.device_num):
#     print(state[i])
