#!
import numpy as np
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader


def numpy_to_dataloader(x: np.ndarray, y: np.ndarray = None, **kwargs) -> DataLoader:
    """
    Example:
    data_loader = numpy_to_dataloader(x, y, batch_size=batch_size)
    """
    if y is None:
        return DataLoader(TensorDataset(Tensor(x)),  **kwargs)
    else:
        return DataLoader(TensorDataset(Tensor(x), Tensor(y)),  **kwargs)