from helical.models.base_models import HelicalRNAModel
import logging
import numpy as np
from anndata import AnnData
import torch
from typing import Union, List, Optional, Dict
from torch.utils.data import DataLoader
from helical.models.stack.stack_config import StackConfig
import pandas as pd
from tqdm.auto import tqdm
import os

# Stack specific imports (assumes stack is installed)
try:
    from stack.model_loading import load_model_from_checkpoint
    from stack.data.training.datasets import load_gene_list
    # Note: we might need more internal imports from stack for data processing
except ImportError:
    logging.getLogger(__name__).warning("Arc-Stack package not found. Please install it with 'pip install -e /path/to/arc-stack'")

LOGGER = logging.getLogger(__name__)

class Stack(HelicalRNAModel):
    """Stack Model Wrapper.

    Stack is a large-scale encoder-decoder foundation model for single-cell biology
    that enables in-context learning for zero-shot perturbation prediction.

    Example
    -------
    ```python
    from helical.models.stack import Stack, StackConfig
    import anndata as ad

    stack_config = StackConfig(
        checkpoint_path="path/to/checkpoint.ckpt",
        genelist_path="path/to/hvg.pkl"
    )
    model = Stack(configurer=stack_config)

    # Load data
    adata = ad.read_h5ad("data.h5ad")
    
    # Get embeddings
    embeddings = model.get_embeddings(adata)
    ```
    """

    def __init__(self, configurer: StackConfig) -> None:
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config
        
        device_str = self.config["device"]
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        LOGGER.info(f"Loading Stack model from {self.config['checkpoint_path']}...")
        
        self.model = load_model_from_checkpoint(
            checkpoint_path=self.config["checkpoint_path"],
            model_class=self.config["model_class"],
            device=self.device
        )
        
        self.genelist = load_gene_list(self.config["genelist_path"])
        LOGGER.info(f"Loaded genelist with {len(self.genelist)} genes.")

    def process_data(self, adata: AnnData) -> AnnData:
        """Processes data for Stack. 
        Note: Stack often relies on its own internal alignment logic.
        """
        LOGGER.info("Processing data for Stack.")
        # For stack, we often pass AnnData directly to its embedding/generation methods
        # but we should ensure it has the required genes.
        return adata

    def get_embeddings(
        self,
        adata: Union[AnnData, str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Extract embeddings using Stack's latent representation method."""
        batch_size = batch_size or self.config["batch_size"]
        
        LOGGER.info("Extracting embeddings using Stack...")
        
        # model.get_latent_representation returns (embeddings, dataset_embeddings)
        embeddings, _ = self.model.get_latent_representation(
            adata_path=adata,
            genelist_path=self.config["genelist_path"],
            batch_size=batch_size,
            show_progress=True,
            num_workers=self.config["num_workers"]
        )
        return embeddings

    def predict_perturbation(
        self,
        base_adata: AnnData,
        test_adata: AnnData,
        split_column: str,
        **kwargs
    ) -> Dict[str, AnnData]:
        """Predict perturbation effects using in-context generation."""
        from stack.cli.generation import generate
        
        LOGGER.info("Predicting perturbation effects using Stack In-Context Generation...")
        
        # This wraps the logic from stack.cli.generation.generate
        # Note: generate usually takes paths, so we might need a version that takes objects
        # or temporarily save objects to disk if absolutely necessary, but preferred to use direct calls.
        
        # For now, let's assume we use the generation logic directly
        from stack.cli.generation import _run_incontext_generation
        
        # Simplified wrapper for a single split/generation
        predictions, test_logit = _run_incontext_generation(
            model=self.model,
            base_adata=base_adata,
            test_adata=test_adata,
            genelist_path=self.config["genelist_path"],
            gene_name_col=kwargs.get("gene_name_col"),
            prompt_ratio=kwargs.get("prompt_ratio", 0.25),
            context_ratio=kwargs.get("context_ratio", 0.4),
            context_ratio_min=kwargs.get("context_ratio_min", 0.2),
            mask_rate=kwargs.get("mask_rate", 1.0),
            mode=kwargs.get("mode", "mdm"),
            num_steps=kwargs.get("num_steps", 5),
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            random_seed=kwargs.get("random_seed", 0),
            show_progress=True
        )
        
        pred_adata = AnnData(
            X=predictions,
            obs=test_adata.obs.copy(),
            var=test_adata.var.copy()
        )
        if test_logit is not None:
             pred_adata.obs["gen_logit"] = np.asarray(test_logit)
             
        return {"predicted": pred_adata}
