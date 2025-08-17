import os
import torch
import anndata as ad
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score
from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel
import torch.distributed as dist
import logging
from datetime import datetime
import sys

# --------------------------
# Configuration
# --------------------------
# Set your desired global batch size here - it will be automatically distributed across GPUs
GLOBAL_BATCH_SIZE = 32

# Fine-tuning task type
FINE_TUNING_HEAD = "classification"  # Options: "classification", "regression", "generation", etc.

# Model configuration
MODEL_NAME = "gf-12L-38M-i4096"  # Change this to use different Geneformer models

# --------------------------
# Logging setup
# --------------------------
def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create log filename with current date
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_filename = f"logs/geneformer_finetune_{current_date}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# --------------------------
# Distributed setup
# --------------------------
def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# --------------------------
# Main training logic
# --------------------------
def main():
    # Setup logging first
    logger = setup_logging()
    
    if is_main_process():
        logger.info("Starting Geneformer fine-tuning script")
        logger.info(f"Fine-tuning task: {FINE_TUNING_HEAD}")
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    
    setup_ddp()

    # Configuration
    model_name = MODEL_NAME
    
    # Calculate per-GPU batch size
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    per_gpu_batch_size = GLOBAL_BATCH_SIZE // world_size
    
    if is_main_process():
        logger.info(f"Global batch size: {GLOBAL_BATCH_SIZE}")
        logger.info(f"Number of GPUs: {world_size}")
        logger.info(f"Per-GPU batch size: {per_gpu_batch_size}")
        logger.info("Downloading dataset...")
        print("Downloading dataset...")
    
    file_path = hf_hub_download(
        repo_id="helical-ai/yolksac_human",
        filename="data/17_04_24_YolkSacRaw_F158_WE_annots.h5ad",
        repo_type="dataset"
    )
    ann_data = ad.read_h5ad(file_path)
    if is_main_process():
        logger.info(f"Dataset loaded from: {file_path}")
        logger.info(f"Dataset shape: {ann_data.shape}")
        logger.info(f"Columns in obs: {list(ann_data.obs.columns)}")
        print("Columns in obs:", ann_data.obs.columns)
        print(ann_data.obs.head())

    # Step 2: Map labels
    cell_types = list(ann_data.obs["LVL3"])
    label_set = sorted(set(cell_types))
    class_id_dict = {c: i for i, c in enumerate(label_set)}
    
    if is_main_process():
        logger.info(f"Number of unique cell types: {len(label_set)}")
        logger.info(f"Cell types: {label_set}")

    def classes_to_ids(example):
        example["cell_types"] = class_id_dict[example["cell_types"]]
        return example

    # Step 3: Model + config
    device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    if is_main_process():
        logger.info(f"Using device: {device}")
        print(f"Using device: {device}")

    geneformer_config = GeneformerConfig(
        model_name=model_name,
        batch_size=per_gpu_batch_size,  # Use calculated per-GPU batch size
        device=device
    )
    geneformer_fine_tune = GeneformerFineTuningModel(
        geneformer_config=geneformer_config,
        fine_tuning_head=FINE_TUNING_HEAD,
        output_size=len(label_set)
    )
    
    if is_main_process():
        logger.info(f"Model initialized: {model_name}")
        logger.info(f"Fine-tuning head: {FINE_TUNING_HEAD}")
        logger.info(f"Per-GPU batch size: {per_gpu_batch_size}")
        logger.info(f"Effective global batch size: {per_gpu_batch_size * world_size}")

    # Step 4: Process dataset
    if is_main_process():
        logger.info("Processing dataset...")
    dataset = geneformer_fine_tune.process_data(ann_data)
    dataset = dataset.add_column('cell_types', cell_types)
    dataset = dataset.map(classes_to_ids, num_proc=4)
    
    if is_main_process():
        logger.info(f"Dataset processed. Total samples: {len(dataset)}")

    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    if is_main_process():
        logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(eval_dataset)}")

    # Step 5: Evaluate before fine-tuning (only on main process)
    if is_main_process():
        logger.info(f"Evaluating model before {FINE_TUNING_HEAD} fine-tuning...")
        print(f"\n=== Evaluating BEFORE {FINE_TUNING_HEAD} fine-tuning ===")
        outputs_before = geneformer_fine_tune.get_outputs(eval_dataset)
        preds_before = outputs_before.argmax(axis=1).tolist()
        true_labels = [ex["cell_types"] for ex in eval_dataset]
        acc_before = accuracy_score(true_labels, preds_before)
        logger.info(f"Pre-fine-tune accuracy: {acc_before:.4f}")
        print(f"Pre-Fine-Tune Accuracy: {acc_before:.4f}")

    # Step 6: Train (DDP automatically handled by torchrun)
    if is_main_process():
        logger.info(f"Starting {FINE_TUNING_HEAD} fine-tuning...")
    
    # Check if model already exists
    model_save_path = f"./custom_finetuned/{model_name}_finetuned_{FINE_TUNING_HEAD}_full"
    model_exists = os.path.exists(model_save_path)
    
    if model_exists and is_main_process():
        logger.info(f"Fine-tuned {FINE_TUNING_HEAD} model already exists at: {model_save_path}")
        print(f"\n=== {FINE_TUNING_HEAD.title()} model already exists at: {model_save_path} ===")
        print("Loading existing model for evaluation...")
        logger.info("Loading existing model for evaluation...")
        
        # Load the existing model
        geneformer_fine_tune.load_model(model_save_path)
        logger.info("Existing model loaded successfully")
        
    else:
        # Train the model
        geneformer_fine_tune.train(
            train_dataset=train_dataset,
            label="cell_types"
        )
        
        if is_main_process():
            logger.info(f"{FINE_TUNING_HEAD.title()} fine-tuning completed")
            # Save model
            geneformer_fine_tune.save_model(model_save_path)
            logger.info(f"Model saved to: {model_save_path}")

    # Step 7: Evaluate after fine-tuning (only on main process)
    if is_main_process():
        if model_exists:
            logger.info(f"Evaluating existing fine-tuned {FINE_TUNING_HEAD} model...")
            print(f"\n=== Evaluating EXISTING fine-tuned {FINE_TUNING_HEAD} model ===")
        else:
            logger.info(f"Evaluating model after {FINE_TUNING_HEAD} fine-tuning...")
            print(f"\n=== Evaluating AFTER {FINE_TUNING_HEAD} fine-tuning ===")
            
        outputs_after = geneformer_fine_tune.get_outputs(eval_dataset)
        preds_after = outputs_after.argmax(axis=1).tolist()
        acc_after = accuracy_score(true_labels, preds_after)
        logger.info(f"Post-fine-tune accuracy: {acc_after:.4f}")
        print(f"Post-Fine-Tune Accuracy: {acc_after:.4f}")

        # Log improvement
        improvement = acc_after - acc_before
        logger.info(f"Accuracy improvement: {improvement:.4f} ({improvement*100:.2f}%)")
        print(f"Accuracy improvement: {improvement:.4f} ({improvement*100:.2f}%)")

    cleanup_ddp()
    
    if is_main_process():
        logger.info("Script completed successfully")

if __name__ == "__main__":
    main()
