from typing import Literal, Optional
from helical.constants.paths import CACHE_DIR_HELICAL
from pathlib import Path
import logging

LOGGER = logging.getLogger(__name__)

class StackConfig:
    """Configuration class to use the Stack Model.

    Parameters
    ----------
    checkpoint_path : str
        Path to the Stack model checkpoint (.ckpt or .pth).
    genelist_path : str
        Path to the gene list (.pkl) used during training/finetuning.
    batch_size : int, optional, default=32
        The batch size for inference.
    device : Literal["cpu", "cuda", "auto"], optional, default="auto"
        The device to use.
    num_workers : int, optional, default=4
        Number of workers for data loading.
    model_class : Literal["scShiftAttentionModel", "ICL_FinetunedModel"], optional, default="scShiftAttentionModel"
        The model class to load from the checkpoint.
    
    Returns
    -------
    StackConfig
        The Stack configuration object
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        genelist_path: Optional[str] = None,
        repo_id: str = "arcinstitute/Stack-Large-Aligned",
        batch_size: int = 32,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        num_workers: int = 4,
        model_class: Literal["scShiftAttentionModel", "ICLFinetunedModel", "ICL_FinetunedModel"] = "scShiftAttentionModel",
    ):
        self.config = {
            "checkpoint_path": checkpoint_path,
            "genelist_path": genelist_path,
            "repo_id": repo_id,
            "batch_size": batch_size,
            "device": device,
            "num_workers": num_workers,
            "model_class": model_class,
        }
