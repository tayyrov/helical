from helical.models.base_models import HelicalRNAModel
import logging
import numpy as np
import anndata as ad
from anndata import AnnData
import torch
from typing import Union, List, Optional, Dict
from torch.utils.data import DataLoader
from helical.models.stack.stack_config import StackConfig
import pandas as pd
from tqdm.auto import tqdm
import os
from pathlib import Path

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

        # Handle auto-download for checkpoints
        checkpoint_path = self.config.get("checkpoint_path")
        genelist_path = self.config.get("genelist_path")
        repo_id = self.config.get("repo_id")

        if (not checkpoint_path or not os.path.exists(checkpoint_path)) or \
           (not genelist_path or not os.path.exists(genelist_path)):
            if repo_id:
                LOGGER.info(f"Checkpoint or genelist missing. Attempting to download from Hugging Face repo: {repo_id}...")
                try:
                    from huggingface_hub import snapshot_download
                    from helical.constants.paths import CACHE_DIR_HELICAL
                    
                    # Use a model-specific cache directory
                    model_dir = Path(CACHE_DIR_HELICAL) / "stack" / repo_id.replace("/", "_")
                    model_dir.mkdir(parents=True, exist_ok=True)
                    
                    download_path = snapshot_download(
                        repo_id=repo_id,
                        local_dir=model_dir,
                        local_dir_use_symlinks=False
                    )
                    
                    # Update paths if they weren't provided or didn't exist
                    # Search for standard filenames if not explicitly hit
                    possible_ckpt = Path(download_path) / "bc_large.ckpt"
                    possible_genelist = Path(download_path) / "basecount_1000per_15000max.pkl"
                    
                    if not checkpoint_path or not os.path.exists(checkpoint_path):
                        if possible_ckpt.exists():
                            checkpoint_path = str(possible_ckpt)
                        else:
                            # Fallback: find first .ckpt
                            ckpts = list(Path(download_path).glob("*.ckpt"))
                            if ckpts:
                                checkpoint_path = str(ckpts[0])
                    
                    if not genelist_path or not os.path.exists(genelist_path):
                        if possible_genelist.exists():
                            genelist_path = str(possible_genelist)
                        else:
                            # Fallback: find first .pkl
                            pkls = list(Path(download_path).glob("*.pkl"))
                            if pkls:
                                genelist_path = str(pkls[0])
                    
                    if not checkpoint_path or not os.path.exists(checkpoint_path):
                        raise FileNotFoundError(f"Could not find a .ckpt file in the downloaded repository: {download_path}")
                    if not genelist_path or not os.path.exists(genelist_path):
                        raise FileNotFoundError(f"Could not find a .pkl genelist in the downloaded repository: {download_path}")
                    
                    self.config["checkpoint_path"] = checkpoint_path
                    self.config["genelist_path"] = genelist_path
                    LOGGER.info(f"Using downloaded checkpoint: {checkpoint_path}")
                    LOGGER.info(f"Using downloaded genelist: {genelist_path}")
                    
                except ImportError:
                    LOGGER.error("huggingface_hub is required for auto-downloading Stack models. Please install it or provide local paths.")
                    raise
            else:
                if not checkpoint_path or not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Stack checkpoint not found and no repo_id provided for download: {checkpoint_path}")
                if not genelist_path or not os.path.exists(genelist_path):
                    raise FileNotFoundError(f"Stack genelist not found and no repo_id provided for download: {genelist_path}")

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
        
        # Force organism column to Homo sapiens for Stack validation
        if isinstance(adata, AnnData):
            adata.obs["organism"] = "Homo sapiens"
            # Ensure unique obs_names to suppress anndata warnings during internal concat
            import uuid
            prefix = str(uuid.uuid4())[:8]
            adata.obs_names = [f"emb_{prefix}_{i}" for i in range(adata.n_obs)]
        
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

    def get_stable_embeddings(
        self,
        adata: Union[AnnData, str],
        context_adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Extract embeddings using a stable context anchor to prevent neighborhood drift.
        
        If context_adata is provided, test cells will be embedded in the context of these cells.
        Otherwise, it falls back to standard get_embeddings with local context.
        """
        if context_adata is None:
            return self.get_embeddings(adata, batch_size=batch_size)
            
        batch_size = batch_size or self.config["batch_size"]
        
        # Force organism column to Homo sapiens for Stack validation
        import uuid
        prefix_a = f"test_{str(uuid.uuid4())[:8]}"
        prefix_c = f"ctx_{str(uuid.uuid4())[:8]}"
        
        for a, prefix in zip([adata, context_adata], [prefix_a, prefix_c]):
            if isinstance(a, AnnData):
                if not a.obs_names.is_unique:
                    a.obs_names_make_unique()
                a.obs["organism"] = "Homo sapiens"
                a.obs_names = [f"{prefix}_{i}" for i in range(a.n_obs)]
                
        LOGGER.info(f"Extracting STABLE embeddings using {context_adata.n_obs} context cells...")
        
        # Call get_incontext_prediction from the underlying stack model
        embeddings = self.model.get_incontext_prediction(
            base_adata_or_path=context_adata,
            test_adata_or_path=adata,
            genelist_path=self.config.get("genelist_path"),
            prompt_ratio=kwargs.get("prompt_ratio", 0.25),
            context_ratio=kwargs.get("context_ratio", 0.40),
            mode='latent',
            batch_size=batch_size,
            num_workers=self.config.get("num_workers", 0),
            random_seed=kwargs.get("random_seed", 0),
            show_progress=False
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
        
        # Ensure organism column exists for both adatas
        for a in [base_adata, test_adata]:
            if "organism" not in a.obs:
                a.obs["organism"] = "Homo sapiens"

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
