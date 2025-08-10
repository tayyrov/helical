import os
import torch
import anndata as ad
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score
from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel
import torch.distributed as dist

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
    setup_ddp()

    model_name = "gf-12L-38M-i4096"

    # Step 1: Load data
    if is_main_process():
        print("Downloading dataset...")
    file_path = hf_hub_download(
        repo_id="helical-ai/yolksac_human",
        filename="data/17_04_24_YolkSacRaw_F158_WE_annots.h5ad",
        repo_type="dataset"
    )
    ann_data = ad.read_h5ad(file_path)
    if is_main_process():
        print("Columns in obs:", ann_data.obs.columns)
        print(ann_data.obs.head())

    # Step 2: Map labels
    cell_types = list(ann_data.obs["LVL3"])
    label_set = sorted(set(cell_types))
    class_id_dict = {c: i for i, c in enumerate(label_set)}

    def classes_to_ids(example):
        example["cell_types"] = class_id_dict[example["cell_types"]]
        return example

    # Step 3: Model + config
    device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    if is_main_process():
        print(f"Using device: {device}")

    geneformer_config = GeneformerConfig(
        model_name=model_name,
        batch_size=32,
        device=device
    )
    geneformer_fine_tune = GeneformerFineTuningModel(
        geneformer_config=geneformer_config,
        fine_tuning_head="classification",
        output_size=len(label_set)
    )

    # Step 4: Process dataset
    dataset = geneformer_fine_tune.process_data(ann_data)
    dataset = dataset.add_column('cell_types', cell_types)
    dataset = dataset.map(classes_to_ids, num_proc=4)

    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Step 5: Evaluate before fine-tuning (only on main process)
    if is_main_process():
        print("\n=== Evaluating BEFORE fine-tuning ===")
        outputs_before = geneformer_fine_tune.get_outputs(eval_dataset)
        preds_before = outputs_before.argmax(axis=1).tolist()
        true_labels = [ex["cell_types"] for ex in eval_dataset]
        acc_before = accuracy_score(true_labels, preds_before)
        print(f"Pre-Fine-Tune Accuracy: {acc_before:.4f}")

    # Step 6: Train (DDP automatically handled by torchrun)
    geneformer_fine_tune.train(
        train_dataset=train_dataset,
        label="cell_types"
    )

    # Step 7: Evaluate after fine-tuning (only on main process)
    if is_main_process():
        print("\n=== Evaluating AFTER fine-tuning ===")
        outputs_after = geneformer_fine_tune.get_outputs(eval_dataset)
        preds_after = outputs_after.argmax(axis=1).tolist()
        acc_after = accuracy_score(true_labels, preds_after)
        print(f"Post-Fine-Tune Accuracy: {acc_after:.4f}")

        # Save model
        model_save_path = f"./custom_finetuned/{model_name}_finetuned_classification_full"
        geneformer_fine_tune.save_model(model_save_path)

    cleanup_ddp()

if __name__ == "__main__":
    main()
