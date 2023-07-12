import torch
import torch.nn as nn
import torch.optim as optim
from Rl_net import actor, critic

learning_start_step_test = 10
learning_start_step = 600
learning_fre = 50
batch_size = 128
batch_size_tets = 10
gamma = 0.9
lr_a = 0.002
lr_c = 0.004
max_grad_norm = 1.0
save_model = 40
save_dir = "models/simple_adversary"
save_fer = 400
tau = 0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Maddpg(object):

    def get_train(self, n_agents, obs_shape_n, action_shape_n):
        actors_cur = [None for _ in range(n_agents)]
        critics_cur = [None for _ in range(n_agents)]
        actors_target = [None for _ in range(n_agents)]
        critics_target = [None for _ in range(n_agents)]
        optimizer_a = [None for _ in range(n_agents)]
        optimizer_c = [None for _ in range(n_agents)]

        for i in range(n_agents):
            actors_cur[i] = actor(obs_shape_n[i], action_shape_n[i]).to(device)
            critics_cur[i] = critic(sum(obs_shape_n),
                                    sum(action_shape_n)).to(device)
            actors_target[i] = actor(obs_shape_n[i],
                                     action_shape_n[i]).to(device)
            critics_target[i] = critic(sum(obs_shape_n),
                                       sum(action_shape_n)).to(device)
            optimizer_a[i] = optim.Adam(actors_cur[i].parameters(), lr=lr_a)
            optimizer_c[i] = optim.Adam(critics_cur[i].parameters(), lr=lr_c)
        actors_tar = self.update_train(actors_cur, actors_target, 1.0)
        critics_tar = self.update_train(critics_cur, critics_target, 1.0)
        return actors_cur, critics_cur, actors_tar, critics_tar, optimizer_a, optimizer_c

    def update_train(self, agents_cur, agents_tar, tau):
        """
        用于更新target网络，
        这个方法不同于直接复制，但结果一样
        out:
        |agents_tar: the agents with new par updated towards agents_current
        """
        for agent_c, agent_t in zip(agents_cur, agents_tar):
            key_list = list(agent_c.state_dict().keys())
            state_dict_t = agent_t.state_dict()
            state_dict_c = agent_c.state_dict()
            for key in key_list:
                state_dict_t[key] = state_dict_c[key] * tau + \
                    (1 - tau) * state_dict_t[key]
            agent_t.load_state_dict(state_dict_t)
        return agents_tar

    def agents_train(self, game_step, update_cnt, memory, obs_size,
                     action_size, actors_cur, actors_tar, critics_cur,
                     critics_tar, optimizers_a, optimizers_c):
        """
        par:
        |input: the data for training
        |output: the data for next update
        """
        # and (game_step-learning_start_step) % learning_fre == 0
        # 训练
        if game_step >= learning_start_step:
            if update_cnt == 0:
                print('\r=start training...' + '' * 100)
            update_cnt += 1

            for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                    enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
                if opt_c == None:
                    continue

                # 随机抽样
                # rew = []
                obs, action, reward, obs_ = memory.sample(
                    batch_size, agent_idx)
                rew = torch.tensor(reward, device=device, dtype=torch.float)
                # print(rew)
                action_cur = torch.from_numpy(action).to(device, torch.float)
                # print('action_batch尺寸\t', action_cur.size())
                # print(action_cur)
                obs_n = torch.from_numpy(obs).to(device, torch.float)
                # print(obs_n.size())
                obs_n_ = torch.from_numpy(obs_).to(device, torch.float)
                with torch.no_grad():
                    action_tar = torch.cat([
                        a_t(obs_n_[:, obs_size[idx][0]:obs_size[idx][1]]).
                        detach() for idx, a_t in enumerate(actors_tar)
                    ],
                                           dim=1)
                q = critic_c(obs_n, action_cur).reshape(-1)  # q
                q_ = critic_t(obs_n_, action_tar).reshape(-1)  # q_
                tar_value = q_ * gamma + rew
                # print(tar_value)
                loss_c = nn.SmoothL1Loss()(q, tar_value)
                opt_c.zero_grad()
                loss_c.backward(retain_graph=True)
                # nn.utils.clip_grad_norm_(critic_c.parameters(), max_grad_norm)
                opt_c.step()

                # update Actor
                # There is no need to cal other agent's action
                # print(obs_n_[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]])
                model_out = actor_c(
                    obs_n_[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]])
                # update the action of this agent
                action_cur[:, action_size[agent_idx][0]:action_size[agent_idx]
                           [1]] = model_out
                # print(action_cur)
                # print(len(action_cur), len(action_cur[0]))
                # loss_pse = torch.mean(torch.pow(policy_c_new, 2))

                actor_loss = critic_c(obs_n, action_cur)
                loss_a = -torch.mean(actor_loss)
                # loss_a = torch.mul(-1, torch.mean(critic_c(obs_n, action_cur)))
                # print('没有噪声影响loss:\t', loss_pse, '\t有噪声影响loss:\t', loss_a)
                opt_a.zero_grad()
                # loss_t = loss_a
                loss_a.backward(retain_graph=True)
                #
                # nn.utils.clip_grad_norm_(actor_c.parameters(), max_grad_norm)
                opt_a.step()

            # update the tar par
            actors_tar = self.update_train(actors_cur, actors_tar, tau)
            critics_tar = self.update_train(critics_cur, critics_tar, tau)
        return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar
