from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel
import anndata as ad
from huggingface_hub import hf_hub_download

# Step 1: Download the .h5ad file
file_path = hf_hub_download(
    repo_id="helical-ai/yolksac_human",
    filename="data/17_04_24_YolkSacRaw_F158_WE_annots.h5ad",
    repo_type="dataset"
)

# Step 2: Load the AnnData object
ann_data = ad.read_h5ad(file_path)

print(ann_data.obs.columns)
print(ann_data.obs.head())

# Get the column for fine-tuning
cell_types = list(ann_data.obs["LVL3"])
label_set = set(cell_types)

# Create a GeneformerConfig object
geneformer_config = GeneformerConfig(model_name="gf-12L-95M-i4096", batch_size=10)

# Create a GeneformerFineTuningModel object
geneformer_fine_tune = GeneformerFineTuningModel(geneformer_config=geneformer_config, fine_tuning_head="classification", output_size=len(label_set))

# Process the data
dataset = geneformer_fine_tune.process_data(ann_data[:10])

# Add column to the dataset
dataset = dataset.add_column('cell_types', cell_types[:10])

# Create a dictionary to map cell types to ids
class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))

def classes_to_ids(example):
    example["cell_types"] = class_id_dict[example["cell_types"]]
    return example

# Convert cell types to ids
dataset = dataset.map(classes_to_ids, num_proc=1)

# Fine-tune the model
geneformer_fine_tune.train(train_dataset=dataset, label="cell_types")

# Get logits from the fine-tuned model
outputs = geneformer_fine_tune.get_outputs(dataset)
print(outputs[:10])

# Get embeddings from the fine-tuned model
embeddings = geneformer_fine_tune.get_embeddings(dataset)
print(embeddings[:10])
