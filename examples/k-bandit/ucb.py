import numpy as np
from kbandit import KBandit
from scipy.stats import beta
from solver import Solver


class UCB(Solver):
    """UCB算法,继承Solver类."""

    def __init__(
        self,
        kbandit: KBandit,
        coef: float,
        init_prob: float = 1.0,
    ) -> None:
        super(UCB, self).__init__(kbandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.num_bandit)
        self.coef = coef

    def estimated_probs(self):
        return self.estimates

    def get_current_action(self) -> int:
        # 计算上置信界
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))
        # 选出上置信界最大的拉杆
        action = np.argmax(ucb)
        return action

    def run_one_step(self) -> None:
        self.total_count += 1
        action = self.get_current_action()
        reward = self.kbandit.step(action)
        self.estimates[action] += (1.0 / (self.counts[action] + 1) *
                                   (reward - self.estimates[action]))

        return action


class BayesianUCB(Solver):
    """Assuming Beta prior."""

    def __init__(self,
                 kbandit: KBandit,
                 coef: float,
                 init_a: float = 1,
                 init_b: float = 1):
        super(BayesianUCB, self).__init__(kbandit)
        self.coef = coef
        # 列表,表示每根拉杆奖励为1的次数
        self._a = np.array([init_a] * self.num_bandit)
        # 列表,表示每根拉杆奖励为0的次数
        self._b = np.array([init_b] * self.num_bandit)
        self.estimates = np.array([0] * self.num_bandit)

    def estimated_probs(self):
        return self.estimates

    def get_current_action(self):
        # ucb
        ucb = self._a / (self._a + self._b) + beta.std(self._a,
                                                       self._b) * self.coef
        # 计算上置信界
        action = np.argmax(ucb)  # 选出上置信界最大的拉杆
        return action

    def run_one_step(self):
        action = self.get_current_action()
        reward = self.kbandit.step(action)
        # 更新Beta分布的第一个参数
        self._a[action] += reward
        # 更新Beta分布的第二个参数
        self._b[action] += 1 - reward
        self.estimates = self._a / (self._a + self._b)
        return action
