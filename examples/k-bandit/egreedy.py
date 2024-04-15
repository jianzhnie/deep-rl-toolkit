import numpy as np
from kbandit import KBandit
from solver import Solver


class EpsilonGreedy(Solver):
    """epsilon贪婪算法,继承Solver类.
    Args:
        - eps (float): the probability to explore at each time step.
        - init_prob (float): default to be 1.0; optimistic initialization
    """

    def __init__(
        self,
        kbandit: KBandit,
        epsilon: float = 0.01,
        init_prob: float = 1.0,
    ) -> None:
        super(EpsilonGreedy, self).__init__(kbandit)
        assert 0.0 <= epsilon <= 1.0
        self.epsilon = epsilon

        # 初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.num_bandit)

    def estimated_probs(self):
        return self.estimates

    def get_current_action(self):
        if np.random.random() < self.epsilon:
            # 随机选择一根拉杆
            action = np.random.randint(0, self.num_bandit)
        else:
            # 选择期望奖励估值最大的拉杆
            action = np.argmax(self.estimates)
        return action

    def run_one_step(self) -> None:
        action = self.get_current_action()
        reward = self.kbandit.step(action)
        self.estimates[action] += (1.0 / (self.counts[action] + 1) *
                                   (reward - self.estimates[action]))
        return action


class DecayEpsilonGreedy(Solver):
    """epsilon值 随时间衰减的epsilon-贪婪算法,继承Solver类.
    Args:
        - eps (float): the probability to explore at each time step.
        - init_prob (float): default to be 1.0; optimistic initialization
    """

    def __init__(
        self,
        kbandit: KBandit,
        epsilon: float = 1.0,
        init_prob: float = 1.0,
    ):
        super(DecayEpsilonGreedy, self).__init__(kbandit)
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.num_bandit)
        self.total_count = 0

    def estimated_probs(self) -> np.array:
        return self.estimates

    def get_current_action(self) -> int:
        self.epsilon = 1.0 / self.total_count
        # epsilon值 随时间衰减
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.num_bandit)
        else:
            action = np.argmax(self.estimates)
        return action

    def run_one_step(self) -> None:
        self.total_count += 1
        action = self.get_current_action()
        reward = self.kbandit.step(action)
        self.estimates[action] += (1.0 / (self.counts[action] + 1) *
                                   (reward - self.estimates[action]))
        return action
