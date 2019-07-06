# -*- coding:utf-8 -*-
""" 表格式的智能体, 用数组存储所有信息
"""
import numpy as np


class TableAgent(object):
    """ 表格式智能体类,智能体中需要存储以下数据
    1. 策略π,表示每一步的行动选择
    2. 状态行动价值函数q(s,a),表示状态s下每一步a的价值
    3. 状态的价值函数v(s),利用贝尔曼公式计算的该状态的价值
    4. 状态转移函数p(a,s1,s2),状态s1下执行行动a使状态转移到s2的概率
    5. 每个状态的奖励r,由环境决定
    """

    def __init__(self, env):
        self._s_len = env.observation_space.n
        self._a_len = env.action_space.n
        self.r = [env.reward(s) for s in range(self._s_len)]
        self.pi = np.array([0 for s in range(self._s_len)])
        self.p = np.zeros([self._a_len, self._s_len, self._a_len])
        ladder_move = np.vectorize(lambda x: env.ladders[x] if x in env.ladders else x)

        for i, dice in enumerate(env.dices):
            prob = 1 / dice
            for src in range(1, 100):
                step = np.arange(dice)
                step += src
                step = np.piecewise(step, [step>100, step<=100], [lambda x: 200-x, lambda x:x])


