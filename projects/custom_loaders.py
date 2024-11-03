from typing import Callable, Optional, Union
from collections import OrderedDict

import torch
from mmcv.runner import get_dist_info
from mmcv.runner.checkpoint import CheckpointLoader, load_url


@CheckpointLoader.register_scheme(prefixes=('https://files.clear.ml/', 'http://files.clear.ml/'))
def load_from_clearml(
        filename: str,
        map_location: Union[str, Callable, None] = None,
        model_dir: Optional[str] = None) -> Union[dict, OrderedDict]:
    """load checkpoint from ClearML, with authentication and caching. In distributed
    setting, this function only download checkpoint at local rank 0.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.
        model_dir (str, optional): directory in which to save the object,
            Default: None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    try:
        from clearml import StorageManager
    except ImportError:
        raise ImportError(
            'Please install clearml to load checkpoint from ClearML with authentication and caching.')

    rank, world_size = get_dist_info()
    if rank == 0:
        checkpoint_fpath = StorageManager.get_local_copy(filename)
        checkpoint = torch.load(checkpoint_fpath, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint_fpath = StorageManager.get_local_copy(filename)
            checkpoint = torch.load(checkpoint_fpath, map_location=map_location)
    return checkpoint
