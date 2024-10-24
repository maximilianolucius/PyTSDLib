import logging
import sys
from typing import Tuple, Type, Dict, Any, Optional

from torch.utils.data import DataLoader

from data_provider.uea import collate_fn
from data_provider.data_loader import (
    Dataset_minute,
    Dataset_Custom,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Mapping of dataset names to their corresponding classes
DATASET_MAPPING: Dict[str, Type] = {
    'Fx5M': Dataset_minute,
    'custom': Dataset_Custom,
    'FxM5': Dataset_minute,
    # Add other datasets here as needed
}

def get_dataset_class(dataset_name: str) -> Type:
    """
    Retrieve the dataset class based on the dataset name.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Type: The corresponding dataset class.

    Raises:
        ValueError: If the dataset name is not supported.
    """
    dataset_class = DATASET_MAPPING.get(dataset_name)
    if dataset_class is None:
        logger.error(f"Unsupported dataset: {dataset_name}. Please check the DATASET_MAPPING.")
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    logger.debug(f"Selected dataset class: {dataset_class.__name__} for dataset: {dataset_name}")
    return dataset_class

def create_data_loader(
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
    collate: Optional[Any] = None
) -> DataLoader:
    """
    Create a DataLoader for the given dataset.

    Args:
        dataset (Any): The dataset object.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses for data loading.
        drop_last (bool): Whether to drop the last incomplete batch.
        collate (Optional[Any]): Custom collate function.

    Returns:
        DataLoader: Configured DataLoader instance.
    """
    if collate:
        logger.debug("Using custom collate function.")
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate
    )
    logger.debug(f"DataLoader created with batch_size={batch_size}, shuffle={shuffle}, "
                 f"num_workers={num_workers}, drop_last={drop_last}")
    return data_loader

def configure_dataset(
    dataset_class: Type,
    args: Any,
    flag: str,
    timeenc: int,
    freq: str
) -> Any:
    """
    Configure and instantiate the dataset based on the task and dataset type.

    Args:
        dataset_class (Type): The dataset class to instantiate.
        args (Any): Parsed command-line arguments.
        flag (str): Indicates the dataset split ('train', 'val', 'test').
        timeenc (int): Time encoding flag.
        freq (str): Frequency of the data.

    Returns:
        Any: Instantiated dataset object.
    """
    dataset_kwargs = {
        'args': args,
        'root_path': args.root_path,
        'flag': flag,
    }

    if args.task_name == 'classification':
            # Add any specific arguments for classification if needed
            pass
    else:
        dataset_kwargs.update({
            'data_path': args.data_path,
            'size': [args.seq_len, args.label_len, args.pred_len],
            'features': args.features,
            'target': args.target,
            'timeenc': timeenc,
            'freq': freq,
            'seasonal_patterns': args.seasonal_patterns,
        })

    dataset = dataset_class(**dataset_kwargs)
    logger.info(f"Initialized dataset '{dataset_class.__name__}' with flag='{flag}' and length={len(dataset)}")
    return dataset

def data_provider(args: Any, flag: str) -> Tuple[Any, DataLoader]:
    """
    Provide the dataset and dataloader based on the task and dataset type.

    Args:
        args (Any): Parsed command-line arguments.
        flag (str): Indicates the dataset split ('train', 'val', 'test').

    Returns:
        Tuple[Any, DataLoader]: The dataset and its corresponding DataLoader.
    """
    try:
        dataset_class = get_dataset_class(args.data)
    except ValueError as e:
        logger.exception("Failed to get dataset class.")
        raise e

    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = not flag.lower() == 'test'
    drop_last = False  # Default value; may be overridden based on task
    batch_size = args.batch_size
    num_workers = args.num_workers
    freq = args.freq

    # Adjust drop_last based on task and dataset
    if args.task_name == 'classification':
        drop_last = False
    else:
        drop_last = True

    # Configure dataset and dataloader based on task
    if args.task_name == 'classification':
        dataset = configure_dataset(dataset_class, args, flag, timeenc, freq)
        data_loader = create_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            drop_last=drop_last,
            collate=lambda x: collate_fn(x, max_len=args.seq_len)
        )
    else:
        dataset = configure_dataset(dataset_class, args, flag, timeenc, freq)
        data_loader = create_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            drop_last=drop_last
        )

    logger.info(f"DataLoader for flag='{flag}' created with {len(dataset)} samples.")
    return dataset, data_loader
