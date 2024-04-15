from typing import List, Union

import numpy as np


class KBandit(object):
    """伯努利多臂老虎机,输入K表示拉杆个数."""

    def __init__(self,
                 num_bandit: int,
                 probs=Union[List[float], np.array]) -> None:
        self.num_bandit = num_bandit
        if probs is not None:
            assert len(probs) == num_bandit
            self.probs = probs
        else:
            # 随机生成K个0～1的数,作为拉动每根拉杆的获奖概率
            self.probs = np.random.uniform(size=num_bandit)

        # 获奖概率最大的拉杆
        self.best_idx = np.argmax(self.probs)
        # 最大的获奖概率
        self.best_prob = self.probs[self.best_idx]

    def step(self, action: int) -> float:
        assert action < self.num_bandit
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未获奖）
        if np.random.rand() < self.probs[action]:
            reward = 1
        else:
            reward = 0
        return reward
