from typing import Any

from rltoolkit.cleanrl.rl_args import RLArguments


class BaseAgent:
    """The basic agent of deep-rl-toolkit. This is an abstract base class for
    all DRL agents.

    Args:
        args (RLArguments): Configuration object for the agent.
    """

    def __init__(self, args: RLArguments) -> None:
        self.args = args

    def get_action(self, *args: Any, **kwargs: Any) -> Any:
        """Return an action with noise when given the observation of the
        environment.

        This function is typically used during training to perform exploration by adding noise to the action.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Action to be taken in the environment.
        """
        raise NotImplementedError

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Predict an estimated Q value when given the observation of the
        environment.

        This function is often used during evaluation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Estimated Q value or action.
        """
        raise NotImplementedError

    def learn(self, *args: Any, **kwargs: Any) -> Any:
        """The training interface for the agent.

        This function is used during the training stage.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Result of the learning step, typically a loss value.
        """
        raise NotImplementedError

    def save_model(self, save_dir: str) -> None:
        """Save the model to the specified directory.

        Args:
            save_dir (str): Directory to save the model.
        """
        raise NotImplementedError

    def load_model(self, save_dir: str) -> None:
        """Load the model from the specified directory.

        Args:
            save_dir (str): Directory to load the model from.
        """
        raise NotImplementedError
