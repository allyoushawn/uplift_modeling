#!/bin/bash

# T-learner Uplift Modeling Runner
# Usage: ./run_t_learner.sh [data_folder]

set -e  # Exit on any error

# Default data folder
DATA_FOLDER=${1:-"data"}

echo "=========================================="
echo "Running T-learner Uplift Modeling"
echo "=========================================="
echo "Data folder: $DATA_FOLDER"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: Virtual environment not detected."
    echo "Please activate your virtual environment first:"
    echo "  source venv/bin/activate"
    echo ""
fi

# Check if data folder exists
if [[ ! -d "$DATA_FOLDER" ]]; then
    echo "Error: Data folder '$DATA_FOLDER' does not exist."
    echo "Please run the data preparation script first:"
    echo "  python prepare_criteo_data.py --output_dir $DATA_FOLDER"
    exit 1
fi

# Check if required data files exist
required_files=("train_x.npy" "train_y.npy" "train_t.npy" "val_x.npy" "val_y.npy" "val_t.npy" "test_x.npy" "test_y.npy" "test_t.npy")
for file in "${required_files[@]}"; do
    if [[ ! -f "$DATA_FOLDER/$file" ]]; then
        echo "Error: Required file '$DATA_FOLDER/$file' not found."
        echo "Please run the data preparation script first:"
        echo "  python prepare_criteo_data.py --output_dir $DATA_FOLDER"
        exit 1
    fi
done

echo "All required data files found. Starting T-learner..."
echo ""

# Run the T-learner script
python t_learner/models/t_learner/run_t_learner.py --data_folder "$DATA_FOLDER"

echo ""
echo "=========================================="
echo "T-learner completed successfully!"
echo "=========================================="
