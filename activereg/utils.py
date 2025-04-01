#!
import numpy as np
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple


def numpy_to_dataloader(x: np.ndarray, y: np.ndarray = None, **kwargs) -> DataLoader:
    """
    Example:
    data_loader = numpy_to_dataloader(x, y, batch_size=batch_size)
    """
    if y is None:
        return DataLoader(TensorDataset(Tensor(x)),  **kwargs)
    else:
        return DataLoader(TensorDataset(Tensor(x), Tensor(y)),  **kwargs)
    

def create_experiment_name(name_set: Tuple) -> str:
    exp_name_list = [str(i) for i in name_set]
    exp_name = exp_name_list[0]+"_".join(exp_name_list[1:])
    return exp_name