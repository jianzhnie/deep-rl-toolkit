import numpy as np
from kbandit import KBandit


class Solver(object):
    """多臂老虎机算法基本框架."""

    def __init__(self, env: KBandit) -> None:
        """bandit (Bandit): the target bandit to solve."""
        self.kbandit = env
        self.num_bandit = env.num_bandit
        # 每根拉杆的尝试次数
        self.counts = np.zeros(self.num_bandit)
        # 当前步的累积懊悔
        self.regret = 0.0
        # 维护一个列表,记录每一步的动作
        self.actions = []
        # 维护一个列表,记录每一步的累积懊悔
        self.regrets = []

    def update_regret(self, action: int) -> None:
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        best_reward = self.kbandit.best_prob
        cur_reward = self.kbandit.probs[action]
        curr_regret = best_reward - cur_reward
        self.regret += curr_regret
        self.regrets.append(self.regret)

    def estimated_probs(self):
        raise NotImplementedError

    def get_current_action(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        raise NotImplementedError

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        raise NotImplementedError

    def run(self, num_steps: int) -> None:
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps):
            action = self.run_one_step()
            self.counts[action] += 1
            self.actions.append(action)
            self.update_regret(action)
