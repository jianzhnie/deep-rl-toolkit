from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from rltoolkit.data import Batch, ReplayBuffer
from rltoolkit.data.utils import to_numpy, to_torch_as
from rltoolkit.policy.base_policy import BasePolicy


class DQNPolicy(BasePolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602.

    Implementation of Double Q-Learning. arXiv:1509.06461.

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here).

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param float target_update_tau: the soft update coefficient for the target, in
        [0, 1]. Default to 1.0, which means hard update. If it is set to a value
        between 0 and 1, it will be a soft update.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool is_double: use double dqn. Default to True.
    :param bool clip_loss_grad: clip the gradient of the loss in accordance
        with nature14236; this amounts to using the Huber loss instead of
        the MSE loss. Default to False.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        target_update_tau: float = 1.0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.eps = 0.0
        assert 0.0 <= discount_factor <= 1.0, 'discount factor should be in [0, 1]'
        self.discount_factor = discount_factor
        assert estimation_step > 0, 'estimation_step should be greater than 0'
        self.estimation_step = estimation_step
        self._target = target_update_freq > 0
        self.target_update_freq = target_update_freq
        assert 0.0 < target_update_tau <= 1.0, 'target_update_tau should be in (0, 1]'
        self.target_update_tau = target_update_tau
        self.num_iters = 0
        if self._target:
            self.target_model = deepcopy(self.model)
            self.target_model.eval()
        self.reward_normalization = reward_normalization
        self.is_double = is_double
        self.clip_loss_grad = clip_loss_grad

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode: bool = True) -> 'DQNPolicy':
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def comput_target_qvalues(self, buffer: ReplayBuffer,
                              indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        result: Batch = self.forward(batch, input='obs_next')
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self.forward(batch,
                                    model='target_model',
                                    input='obs_next').logits
        else:
            target_q = result.logits
        if self.is_double:
            return target_q[np.arange(len(result.act)), result.act]
        else:  # Nature DQN, over estimate
            return target_q.max(dim=1)[0]

    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indices: np.ndarray) -> Batch:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        batch = self.compute_nstep_return(
            batch,
            buffer,
            indices,
            target_q_fn=self.comput_target_qvalues,
            gamma=self.discount_factor,
            n_step=self.estimation_step,
            rew_norm=self.reward_normalization,
        )
        return batch

    def compute_q_value(self, logits: torch.Tensor,
                        mask: Optional[np.ndarray]) -> torch.Tensor:
        """Compute the q value based on the network's raw output and action
        mask."""
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = 'model',
        input: str = 'obs',
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :return: A :class:`~rltoolkit.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, 'obs') else obs
        logits, hidden = model(obs_next, state=state, info=batch.info)
        q_value = self.compute_q_value(logits, getattr(obs, 'mask', None))
        if not hasattr(self, 'max_action_num'):
            self.max_action_num = q_value.shape[1]
        act = to_numpy(q_value.max(dim=1)[1])
        return Batch(logits=logits, act=act, state=hidden)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self.num_iters % self.target_update_freq == 0:
            self.sync_weight(
                src=self.model,
                tgt=self.target_model,
                tau=self.target_update_tau,
            )
        self.optim.zero_grad()
        weight = batch.pop('weight', 1.0)
        q_value = self.forward(batch).logits
        q_value = q_value[np.arange(len(q_value)), batch.act]
        returns = to_torch_as(batch.returns.flatten(), q_value)
        td_error = returns - q_value

        if self.clip_loss_grad:
            y = q_value.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(y, t, reduction='mean')
        else:
            loss = (td_error.pow(2) * weight).mean()

        batch.weight = td_error  # prio-buffer
        loss.backward()
        self.optim.step()
        self.num_iters += 1
        return {'loss': loss.item()}

    def exploration_noise(
        self,
        act: Union[np.ndarray, Batch],
        batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, 'mask'):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act
