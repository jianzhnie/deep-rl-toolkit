import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rltoolkit.agents.rl_args import RLArguments


class BaseAgent:
    """The basic agent of deep-rl-toolkit. It is an abstract class for all DRL
    agents.

    Args:
        args (RLArguments): argsuration object for the agent
    """

    def __init__(self, args: RLArguments) -> None:
        self.args = args

    def sample(self, *args, **kwargs):
        """Return an action with noise when given the observation of the
        environment.

        In general, this function is used in train process as noise is added to
        the action to preform exploration.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Predict an estimated Q value when given the observation of the
        environment.

        It is often used in the evaluation stage.
        """
        raise NotImplementedError

    def learn(self, *args, **kwargs):
        """The training interface for ``Agent``.

        It is often used in the training stage.
        """
        raise NotImplementedError

    def save_model(self, save_dir: str) -> None:
        """Save the model.

        Args:
            save_dir (str): Directory to save the model.
            steps (int): Current training step.
        """
        raise NotImplementedError

    def load_model(self, save_dir: str) -> None:
        """Load the model.

        Args:
            save_dir (str): Directory to load the model from.
            steps (int): Current training step.
        """
        raise NotImplementedError
