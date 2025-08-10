from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel
import anndata as ad
from huggingface_hub import hf_hub_download
import torch
from sklearn.metrics import accuracy_score

# Step 1: Download the .h5ad file
file_path = hf_hub_download(
    repo_id="helical-ai/yolksac_human",
    filename="data/17_04_24_YolkSacRaw_F158_WE_annots.h5ad",
    repo_type="dataset"
)

# Step 2: Load the AnnData object
ann_data = ad.read_h5ad(file_path)

print("Columns in obs:", ann_data.obs.columns)
print(ann_data.obs.head())

# Get cell type labels and unique set
cell_types = list(ann_data.obs["LVL3"])
label_set = sorted(set(cell_types))  # sorted for consistent mapping

# Map cell types to IDs
class_id_dict = {c: i for i, c in enumerate(label_set)}

def classes_to_ids(example):
    example["cell_types"] = class_id_dict[example["cell_types"]]
    return example

# Detect device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create config with device
geneformer_config = GeneformerConfig(model_name="gf-12L-38M-i4096", batch_size=32, device=device)

# Initialize fine-tuning model
geneformer_fine_tune = GeneformerFineTuningModel(
    geneformer_config=geneformer_config,
    fine_tuning_head="classification",
    output_size=len(label_set)
)

# Process the whole dataset (no slicing)
dataset = geneformer_fine_tune.process_data(ann_data)

# Add 'cell_types' column to dataset
dataset = dataset.add_column('cell_types', cell_types)

# Convert classes to IDs
dataset = dataset.map(classes_to_ids, num_proc=4)

# Split dataset into train and eval (e.g., 80/20)
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# Fine-tune the model with multiple epochs
geneformer_fine_tune.train(
    train_dataset=train_dataset,
    label="cell_types"
)

# Save the fine-tuned model
model_save_path = "./geneformer_finetuned"
geneformer_fine_tune.save_model(model_save_path)

# Reload model from saved path (optional)
# geneformer_fine_tune.load_model(model_save_path)

# Evaluate on the eval set
outputs = geneformer_fine_tune.get_outputs(eval_dataset)
preds = outputs.argmax(axis=1).tolist()

# Get true labels from eval dataset
true_labels = [ex["cell_types"] for ex in eval_dataset]

# Calculate accuracy
acc = accuracy_score(true_labels, preds)
print(f"Evaluation Accuracy: {acc:.4f}")

