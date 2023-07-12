import torch
import torch.nn as nn
import torch.nn.functional as F


class abstract_agent(nn.Module):

    def __init__(self):
        super().__init__()

    def act(self, x):
        policy, value = self.forward(x)
        return policy, value


class critic(abstract_agent):
    # state15*15=225，action-->15*4 =60
    def __init__(self, obs_shape, act_shape):
        super().__init__()
        # self.LRelu = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(act_shape + obs_shape, 512)
        self.linear_c2 = nn.Linear(512, 256)
        self.linear_c3 = nn.Linear(256, 64)
        self.linear_c4 = nn.Linear(64, 16)
        self.linear_c = nn.Linear(16, 1)

    def forward(self, obs_input, act_input):
        x_cat = F.relu(self.linear_c1(torch.cat([obs_input, act_input],
                                                dim=1)))
        x = F.relu(self.linear_c2(x_cat))
        x = F.relu(self.linear_c3(x))
        x = F.relu(self.linear_c4(x))
        x = self.linear_c(x)

        return x


class actor(abstract_agent):
    # 输入n_feature --> 15； action--> 4
    def __init__(self, num_input, action_size):
        super().__init__()
        self.linear_a1 = nn.Linear(num_input, 32)
        self.linear_a2 = nn.Linear(32, 16)
        self.linear_a = nn.Linear(16, action_size)

    def forward(self, x):
        x = F.relu(self.linear_a1(x))
        x = F.relu(self.linear_a2(x))
        # model_out = F.softmax(self.linear_a(x), dim=-1)
        model_out = F.relu(self.linear_a(x))
        # u = torch.rand_like(model_out)
        # policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        # # policy = F.relu(model_out - torch.log(-torch.log(u)))
        # if model_original_out:
        #     return model_out, policy
        return model_out
