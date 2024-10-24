import argparse
import logging
from pathlib import Path
import random
import sys

import numpy as np
import torch
from datetime import datetime

from exp.exp_classification import Exp_Classification
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from utils.print_args import print_args

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 2021) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.debug(f"Random seeds set to {seed}")

def valid_date(s: str) -> datetime:
    """
    Validate and parse a date string in YYYY-MM-DD format.

    Args:
        s (str): The date string to validate.

    Returns:
        datetime: Parsed datetime object.

    Raises:
        argparse.ArgumentTypeError: If the date string is not in the correct format.
    """
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = f"Not a valid date: '{s}'. Expected format: YYYY-MM-DD."
        raise argparse.ArgumentTypeError(msg)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='TimesNet - A Time Series Analysis Tool')

    # Basic configuration
    parser.add_argument('--task_name', type=str, required=True,
                        choices=['long_term_forecast', 'short_term_forecast', 'classification'],
                        help='Task name to execute.')
    parser.add_argument('--is_training', type=int, required=True, choices=[0, 1],
                        help='Set to 1 for training, 0 for testing.')
    parser.add_argument('--model_id', type=str, default='test',
                        help='Identifier for the model instance.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['Autoformer', 'Transformer', 'TimesNet', 'TimeMixer', 'NuPIC', 'PatchTST'],
                        help='Model architecture to use.')

    # Data loader configuration
    parser.add_argument('--data', type=str, required=True, default='ETTm1',
                        help='Type of dataset to use.')
    parser.add_argument('--root_path', type=Path, default=Path('./data/ETT/'),
                        help='Root directory of the data files.')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv',
                        help='Name of the data file.')
    parser.add_argument('--features', type=str, default='M',
                        choices=['M', 'S', 'MS'],
                        help='Type of forecasting task: M (multivariate), S (univariate), MS (multivariate to univariate).')
    parser.add_argument('--target', type=str, default='OT',
                        help='Target feature for S or MS tasks.')
    parser.add_argument('--freq', type=str, default='h',
                        help='Frequency for time features encoding (e.g., "h" for hourly).')
    parser.add_argument('--checkpoints', type=Path, default=Path('./checkpoints/'),
                        help='Directory to save model checkpoints.')

    # Forecasting task parameters
    parser.add_argument('--seq_len', type=int, default=96, help='Length of the input sequence.')
    parser.add_argument('--label_len', type=int, default=48, help='Length of the start token.')
    parser.add_argument('--pred_len', type=int, default=96, help='Length of the prediction sequence.')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly',
                        help='Seasonal pattern subset for M4 dataset.')
    parser.add_argument('--inverse', action='store_true', default=False,
                        help='Inverse the output data.')

    # Model definition parameters
    parser.add_argument('--expand', type=int, default=2, help='Expansion factor for Mamba.')
    parser.add_argument('--d_conv', type=int, default=4, help='Convolution kernel size for Mamba.')
    parser.add_argument('--top_k', type=int, default=5, help='Top K for TimesBlock.')
    parser.add_argument('--num_kernels', type=int, default=6, help='Number of kernels for Inception.')
    parser.add_argument('--enc_in', type=int, default=7, help='Encoder input size.')
    parser.add_argument('--dec_in', type=int, default=7, help='Decoder input size.')
    parser.add_argument('--c_out', type=int, default=7, help='Output size.')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers.')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers.')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of the feedforward network.')
    parser.add_argument('--moving_avg', type=int, default=25, help='Window size for moving average.')
    parser.add_argument('--factor', type=int, default=1, help='Attention factor.')
    parser.add_argument('--distil', action='store_false', default=True,
                        help='Disable distillation in the encoder.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--embed', type=str, default='timeF',
                        choices=['timeF', 'fixed', 'learned'],
                        help='Type of time features encoding.')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function.')
    parser.add_argument('--channel_independence', type=int, default=1,
                        choices=[0, 1],
                        help='0: channel dependence, 1: channel independence for FreTS model.')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        choices=['moving_avg', 'dft_decomp'],
                        help='Method of series decomposition.')
    parser.add_argument('--use_norm', type=int, default=1, choices=[0, 1],
                        help='Whether to use normalization (1) or not (0).')
    parser.add_argument('--down_sampling_layers', type=int, default=0,
                        help='Number of down-sampling layers.')
    parser.add_argument('--down_sampling_window', type=int, default=1,
                        help='Down-sampling window size.')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        choices=['avg', 'max', 'conv'],
                        help='Method for down-sampling.')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='Segment length for SegRNN iteration.')

    # Optimization parameters
    parser.add_argument('--num_workers', type=int, default=10, help='Number of data loader workers.')
    parser.add_argument('--itr', type=int, default=1, help='Number of experiment iterations.')
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer.')
    parser.add_argument('--des', type=str, default='test', help='Experiment description.')
    parser.add_argument('--loss', type=str, default='MSE', help='Loss function.')
    parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment strategy.')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use automatic mixed precision training.')

    # GPU configuration
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use GPU.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index.')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='Whether to use multiple GPUs.')
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='Comma-separated list of GPU device IDs.')

    # De-stationary projector parameters
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='Hidden layer dimensions of projector.')
    parser.add_argument('--p_hidden_layers', type=int, default=2,
                        help='Number of hidden layers in projector.')

    # Metrics
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='Use Dynamic Time Warping (DTW) metric.')

    # Data Augmentation parameters
    parser.add_argument('--augmentation_ratio', type=int, default=0,
                        help='Number of times to augment the data.')
    parser.add_argument('--seed', type=int, default=2, help='Randomization seed for augmentation.')
    parser.add_argument('--jitter', action='store_true', default=False,
                        help='Apply jitter augmentation.')
    parser.add_argument('--scaling', action='store_true', default=False,
                        help='Apply scaling augmentation.')
    parser.add_argument('--permutation', action='store_true', default=False,
                        help='Apply equal length permutation augmentation.')
    parser.add_argument('--randompermutation', action='store_true', default=False,
                        help='Apply random length permutation augmentation.')
    parser.add_argument('--magwarp', action='store_true', default=False,
                        help='Apply magnitude warp augmentation.')
    parser.add_argument('--timewarp', action='store_true', default=False,
                        help='Apply time warp augmentation.')
    parser.add_argument('--windowslice', action='store_true', default=False,
                        help='Apply window slice augmentation.')
    parser.add_argument('--windowwarp', action='store_true', default=False,
                        help='Apply window warp augmentation.')
    parser.add_argument('--rotation', action='store_true', default=False,
                        help='Apply rotation augmentation.')
    parser.add_argument('--spawner', action='store_true', default=False,
                        help='Apply SPAWNER augmentation.')
    parser.add_argument('--dtwwarp', action='store_true', default=False,
                        help='Apply DTW warp augmentation.')
    parser.add_argument('--shapedtwwarp', action='store_true', default=False,
                        help='Apply Shape DTW warp augmentation.')
    parser.add_argument('--wdba', action='store_true', default=False,
                        help='Apply Weighted DBA augmentation.')
    parser.add_argument('--discdtw', action='store_true', default=False,
                        help='Apply Discriminative DTW warp augmentation.')
    parser.add_argument('--discsdtw', action='store_true', default=False,
                        help='Apply Discriminative Shape DTW warp augmentation.')
    parser.add_argument('--extra_tag', type=str, default="",
                        help='Additional tag for the experiment.')

    parser.add_argument('--valid_date', type=valid_date, default=None,
                        help='Validation date in YYYY-MM-DD format. Default is None.')

    args = parser.parse_args()
    return args


def configure_gpu(args: argparse.Namespace) -> None:
    """
    Configure GPU settings based on availability and user arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    logger.info(f"GPU Available: {torch.cuda.is_available()}")

    if args.use_gpu:
        if args.use_multi_gpu:
            device_ids = [int(id_.strip()) for id_ in args.devices.split(',')]
            args.device_ids = device_ids
            args.gpu = device_ids[0]
            logger.info(f"Using multiple GPUs: {args.device_ids}")
        else:
            args.gpu = args.gpu
            logger.info(f"Using single GPU: {args.gpu}")
    else:
        logger.info("Using CPU for computation.")


def select_experiment(task_name: str):
    """
    Select the appropriate experiment class based on the task name.

    Args:
        task_name (str): The name of the task.

    Returns:
        Experiment class corresponding to the task.
    """
    task_mapping = {
        'long_term_forecast': Exp_Long_Term_Forecast,
        'short_term_forecast': Exp_Short_Term_Forecast,
        'classification': Exp_Classification
    }
    return task_mapping.get(task_name, Exp_Long_Term_Forecast)


def generate_setting(args: argparse.Namespace, iteration: int) -> str:
    """
    Generate a unique setting string for the experiment.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        iteration (int): Current iteration index.

    Returns:
        str: Formatted setting string.
    """
    setting = (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}_"
        f"sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_"
        f"el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_expand{args.expand}_dc{args.d_conv}_"
        f"fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{iteration}"
    )
    return setting


def main() -> None:
    """
    Main function to execute the experiment based on provided arguments.
    """
    # Set random seeds for reproducibility
    set_random_seeds()

    # Parse command-line arguments
    args = parse_arguments()

    # Configure GPU settings
    configure_gpu(args)

    # Log the parsed arguments
    logger.info("Arguments for the experiment:")
    print_args(args)

    # Select the appropriate experiment class
    Exp = select_experiment(args.task_name)

    if args.is_training:
        for itr in range(args.itr):
            # Generate unique setting identifier
            setting = generate_setting(args, itr)

            logger.info(f">>>>>>> Start Training: {setting} >>>>>>>>>>>>>>>>>>>>")
            exp = Exp(args)  # Initialize experiment
            exp.train(setting)

            logger.info(f">>>>>>> Start Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)

            # Clear GPU cache to free memory
            torch.cuda.empty_cache()
    else:
        # Single test setting
        itr = 0
        setting = generate_setting(args, itr)

        logger.info(f">>>>>>> Start Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<")
        exp = Exp(args)  # Initialize experiment
        exp.test(setting, test=1)

        # Clear GPU cache to free memory
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
