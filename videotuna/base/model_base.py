from pathlib import Path
from typing import Any, Dict, Union

import torch.nn as nn


class ModelBase(nn.Module):
    """
    A base class for all models. This class extends nn.Module from PyTorch and
    provides a structure that all models should follow, including initialization,
    forward pass, and utility methods for saving/loading models, getting model
    configuration, and counting model parameters.
    """

    def __init__(self):
        """
        Initializes the ModelBase class. This method should be overridden in any
        subclass to initialize model-specific components.
        """
        super().__init__()

    def forward(self):
        """
        Defines the forward pass of the model. This method should be implemented
        in any subclass to specify how input data should be processed through the
        network.
        """
        raise NotImplementedError("Please implement the forward method.")

    def save_model(self, path: Union[str, Path]):
        """
        Saves the model to a specified path. This method should be implemented in
        any subclass to define how the model's state is saved.

        Args:
            path (Union[str, Path]): The file path where the model will be saved.
        """
        pass

    def load_model(self, path: Union[str, Path]):
        """
        Loads the model from a specified path. This method should be implemented in
        any subclass to define how the model's state is loaded.

        Args:
            path (Union[str, Path]): The file path from where the model will be loaded.
        """
        pass

    def get_model_config(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing the configuration of the model. This
        method should be implemented in any subclass to provide a way to access
        the model's configuration settings.

        Returns:
            Dict[str, Any]: A dictionary with model configuration details.
        """
        pass

    def get_param_count(self):
        """
        Gets the total number of parameters in the model. This method should be
        implemented in any subclass to provide a way to count the parameters.
        """
        pass
