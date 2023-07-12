import numpy as np
import torch


class OUNoise(object):
    '''Ornstein–Uhlenbeck噪声
    '''

    def __init__(self, n_actions, mu=0.0, theta=0.15, sigma=0.35):
        self.mu = mu  # OU噪声的参数，均值
        self.theta = theta  # OU噪声的参数，回复到平均值的速度
        self.sigma = sigma  # OU噪声的参数，标准差

        self.n_actions = n_actions
        self.reset()

    def reset(self):
        # 把内部状态重新设置为平均值
        self.action = np.ones(self.n_actions) * self.mu

    def __call__(self):
        x = self.action
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.n_actions)
        self.action = x + dx
        return self.action
