#!/bin/bash

# Script to run Geneformer fine-tuning with proper logging
# Usage: ./run_fine_tuning.sh [num_gpus]

# Set default number of GPUs if not specified
NUM_GPUS=${1:-"gpu"}

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
LOG_FILE="./logs/geneformer_finetune_${TIMESTAMP}.log"

if [ "${NUM_GPUS}" = "gpu" ]; then
    echo "Starting Geneformer fine-tuning with auto-detected GPUs"
else
    echo "Starting Geneformer fine-tuning with ${NUM_GPUS} GPUs"
fi
echo "Log file: ${LOG_FILE}"
echo "Timestamp: ${TIMESTAMP}"
echo "=========================================="

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. CUDA may not be properly installed."
    exit 1
fi

# Show GPU status before starting
echo "GPU Status before training:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

echo ""
echo "Starting fine-tuning..."
echo "=========================================="

# Run the fine-tuning script with torchrun and capture all output
torchrun --nproc_per_node=${NUM_GPUS} test_fine_tuning_full.py 2>&1 | tee "${LOG_FILE}"

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "Fine-tuning completed successfully!"
    echo "Log saved to: ${LOG_FILE}"
    
    # Show GPU status after completion
    echo ""
    echo "GPU Status after training:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
    
    
else
    echo "=========================================="
    echo "Fine-tuning failed with exit code ${EXIT_CODE}"
    echo "Check log file for details: ${LOG_FILE}"
    
    
    exit $EXIT_CODE
fi
