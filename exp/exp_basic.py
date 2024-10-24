import logging
import os
from typing import Any, Dict, Type, Tuple

import torch
import torch.nn as nn

from models import (
    Autoformer, Transformer, TimesNet, DLinear, Crossformer,
    iTransformer, TiDE, TimeMixer, TSMixer, SegRNN, SCINet, PAttn, PatchTST
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class Exp_Basic:
    """
    Base Experiment class providing common functionalities for different experiment types.
    """

    def __init__(self, args: Any):
        """
        Initialize the experiment with given arguments.

        Args:
            args (Any): Parsed command-line arguments containing experiment settings.
        """
        self.args = args
        self.model_dict: Dict[str, Type[nn.Module]] = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'DLinear': DLinear,
            'Crossformer': Crossformer,
            'iTransformer': iTransformer,
            'TiDE': TiDE,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'PatchTST': PatchTST
        }

        # Dynamically add 'Mamba' model if specified
        if self.args.model == 'Mamba':
            try:
                from models import Mamba
                self.model_dict['Mamba'] = Mamba
                logger.info("Mamba model added to model_dict.")
            except ImportError as e:
                logger.error("Failed to import Mamba model. Please ensure mamba_ssm is installed.")
                raise ImportError("mamba_ssm is not installed.") from e

        # Acquire device
        self.device = self._acquire_device()

        # Build and move model to device
        self.model = self._build_model().to(self.device)
        logger.info(f"Model {self.args.model} initialized and moved to {self.device}.")

    def _build_model(self) -> nn.Module:
        """
        Build and return the model based on the model name provided in args.

        Returns:
            nn.Module: The constructed model.

        Raises:
            ValueError: If the specified model is not supported.
        """
        model_class = self.model_dict.get(self.args.model)
        if model_class is None:
            logger.error(f"Model {self.args.model} is not supported.")
            raise ValueError(f"Model {self.args.model} is not supported.")

        model = model_class(self.args)
        logger.info(f"Model {self.args.model} built successfully.")
        return model

    def _acquire_device(self) -> torch.device:
        """
        Acquire and configure the computation device (CPU or GPU).

        Returns:
            torch.device: The device to be used for computations.
        """
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                # Set CUDA_VISIBLE_DEVICES environment variable
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                device_ids = [int(id_.strip()) for id_ in self.args.devices.split(',')]
                torch.cuda.set_device(device_ids[0])  # Set the primary GPU
                logger.info(f"Using multiple GPUs: {device_ids}")
                device = torch.device(f'cuda:{device_ids[0]}')
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                device = torch.device(f'cuda:{self.args.gpu}')
                logger.info(f"Using single GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for computations.")
        return device

    def _get_data(self, flag: str):
        """
        Placeholder method to get data. Should be implemented in subclasses.

        Args:
            flag (str): Data split flag ('train', 'val', 'test').

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError("The _get_data method must be implemented in the subclass.")

    def vali(self, *args, **kwargs):
        """
        Placeholder validation method. Should be implemented in subclasses.
        """
        raise NotImplementedError("The vali method must be implemented in the subclass.")

    def train(self, *args, **kwargs):
        """
        Placeholder training method. Should be implemented in subclasses.
        """
        raise NotImplementedError("The train method must be implemented in the subclass.")

    def test(self, *args, **kwargs):
        """
        Placeholder testing method. Should be implemented in subclasses.
        """
        raise NotImplementedError("The test method must be implemented in the subclass.")
