from typing import List

import matplotlib.pyplot as plt
import numpy as np
from egreedy import DecayEpsilonGreedy, EpsilonGreedy
from kbandit import KBandit
from solver import Solver
from ucb import UCB, BayesianUCB


def plot_results(
    solvers: List[Solver],
    solver_names: List[str],
    filename: str = None,
) -> None:
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称.

    Plot the results by multi-armed bandit solvers.
    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str)
        figname (str)
    """
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    bandit: KBandit = solvers[0].kbandit

    fig = plt.figure(figsize=(18, 5))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Sub.fig. 1: Regrets in time.
    for idx, solver in enumerate(solvers):
        ax1.plot(range(len(solver.regrets)),
                 solver.regrets,
                 label=solver_names[idx])

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('nbandit', ls='--', alpha=0.3)

    # # Sub.fig. 2: Probabilities estimated by solvers.
    sorted_indices = sorted(range(bandit.num_bandit),
                            key=lambda x: bandit.probs[x])
    ax2.plot(
        range(bandit.num_bandit),
        [bandit.probs[x] for x in sorted_indices],
        'nbandit--',
        markersize=12,
    )
    for solver in solvers:
        ax2.plot(
            range(bandit.num_bandit),
            [solver.estimates[x] for x in sorted_indices],
            'x',
            markeredgewidth=2,
        )
    ax2.set_xlabel('Actions sorted by ' + r'θ')
    ax2.set_ylabel('Estimated')
    ax2.grid('nbandit', ls='--', alpha=0.3)

    # Sub.fig. 3: Action counts
    for solver in solvers:
        ax3.plot(
            range(bandit.num_bandit),
            np.array(solver.counts) / float(len(solvers[0].regrets)),
            ls='solid',
            lw=2,
        )
    ax3.set_xlabel('Actions')
    ax3.set_ylabel('Frac. # trials')
    ax3.grid('k', ls='--', alpha=0.3)

    plt.savefig(filename)


def experiment(num_bandit: int = 10, total_steps: int = 5000):
    """Run a small experiment on solving a Bernoulli bandit with K slot
    machines, each with a randomly initialized reward probability.

    Args:
        num_bandit (int): number of slot machiens.
        total_steps (int): number of time steps to try.
    """

    bandit = KBandit(num_bandit)
    print('Randomly generated Bernoulli bandit has reward probabilities:\n',
          bandit.probs)
    print('The best machine has index: {} and proba: {}'.format(
        max(range(num_bandit), key=lambda i: bandit.probs[i]),
        max(bandit.probs)))

    test_solvers = [
        EpsilonGreedy(bandit, 0.1),
        DecayEpsilonGreedy(bandit, 1),
        UCB(bandit, coef=1.0),
        BayesianUCB(bandit, 3, 1, 1),
    ]
    names = [
        r'ϵ' + '-Greedy',
        r'ϵ' + '-DecayGreedy',
        'UCB1',
        'Bayesian UCB',
    ]

    for solver in test_solvers:
        solver.run(total_steps)

    plot_results(test_solvers, names,
                 'results_K{}_N{}.png'.format(num_bandit, total_steps))


if __name__ == '__main__':
    experiment(10, 10000)
