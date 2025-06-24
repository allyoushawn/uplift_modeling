#!/bin/bash

# Run DragonNet training
echo "Starting DragonNet training..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set log directory for TensorBoard
LOG_DIR="runs"
mkdir -p "$LOG_DIR"

# Run DragonNet with default parameters
python dragon_net/run_dragon_net.py \
    --data_dir data \
    --batch_size 512 \
    --num_epochs 50 \
    --lr 0.001 \
    --alpha 0.1 \
    --beta 0.1 \
    --hidden_sizes 200,100,100 \
    --dropout 0.1 \
    --log_dir "$LOG_DIR"

echo "DragonNet training completed!" 