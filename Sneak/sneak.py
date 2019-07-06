# -*- coding:utf-8 -*-
""" 蛇棋环境, 根据强化学习精要这个书做的练习
"""
import numpy as np
import gym
from gym.spaces import Discrete


class Sneak(gym.Env):
    """ 蛇棋环境,总共100格,走到终点获胜,
    超出则返回,棋盘中间有各种梯子,投骰子没有
    到达终点记-1分,达到终点奖励100分
    """
    SIZE = 100

    def __init__(self, ladder_num, dices=[3, 6]):
        """ 初始化棋盘设置
        ladder_num : 梯子的个数
        dices : 骰子的投掷方法,每种投法的最大值
        """
        self._ladder_num = ladder_num
        self._dices = dices
        # 把表示梯子的数值对变成字典, 字典的键只能有一个值, 会自动排除多值的情况, 但是会导致梯子个数变少
        self._ladders = dict(np.random.randint(1, self.SIZE, (self._ladder_num, 2)))
        # 定义观测空间和行动空间，根据gym的定义这两个值必须设置
        # Discrete函数返回的观测空间为0-max，这里我们要用的观测是1-100所以这里要+1,0这个观测就不用
        self.observation_space = Discrete(self.SIZE+1)
        self.action_space = Discrete(len(dices))
        # 由于梯子是双向的, 设置梯子的另一个方向
        for k, v in self._ladders.items():
            self._ladders[v] = k
        # 最开始位置在1
        self._pos = 1

    def reset(self):
        """ 重置环境并返回初始化观测值
        """
        self._pos = 1
        return self._pos

    def step(self, action):
        """ 运行一次环境的仿真步, 根据给定的action返回下一时刻的环境
        Args:
            action (object): 智能体的行动

        Returns:
            observation (object): 完成行动后的环境观测值
            reward (float) : 这一步行动的奖励值
            done (bool): 是否完成目标, 如果是的话必须初始化, 否则后一步行动结果未知
            info (dict): 各种辅助信息 (用于调试, 有时候也用来学习这个信息)
        """
        # 先投一次骰子
        step = np.random.randint(1, self._dices[action]+1)
        self._pos += step
        if self._pos == 100:
            return 100, 100, 1, {}
        elif self._pos > 100:
            self._pos = 200 - self._pos
        if self._pos in self._ladders:
            self._pos = self._ladders[self._pos]
        return self._pos, -1, 0, {}

    def reward(self, state):
        """ 这个不是必须的函数
        """
        if state == 100:
            return 100
        else:
            return -1

    def render(self, mode='human'):
        pass
