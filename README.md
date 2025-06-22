# Criteo Uplift Modeling - Data Preparation

This project demonstrates data preparation for uplift modeling using the Criteo dataset.

## Overview

The `prepare_criteo_data.py` script processes the Criteo Uplift v2.1 dataset and splits it into training, validation, and test sets for uplift modeling experiments.

## Features

- **Flexible labeling**: Choose between 'exposure', 'visit', or 'conversion' as the target variable
- **Treatment selection**: Select from 'treatment', 'exposure', or 'visit' as the treatment variable
- **Data sampling**: Option to use a subset of the data for faster experimentation
- **Stratified splitting**: Maintains class distribution across train/validation/test splits

## Usage

### Basic Usage
```bash
python prepare_criteo_data.py
```

### With Custom Parameters
```bash
python prepare_criteo_data.py \
    --label conversion \
    --treatment treatment \
    --input criteo-uplift-v2.1.csv \
    --output_dir data \
    --sample_ratio 0.01
```

### Parameters

- `--label`: Target variable ('exposure', 'visit', 'conversion') - default: 'conversion'
- `--treatment`: Treatment variable ('treatment', 'exposure', 'visit') - default: 'treatment'
- `--input`: Input CSV file path - default: 'criteo-uplift-v2.1.csv'
- `--output_dir`: Output directory for .npy files - default: 'data'
- `--sample_ratio`: Fraction of data to use (0.0 to 1.0) - default: 1.0

## Output

The script generates the following files in the output directory:
- `train_x.npy`, `train_y.npy`, `train_t.npy` - Training data
- `val_x.npy`, `val_y.npy`, `val_t.npy` - Validation data  
- `test_x.npy`, `test_y.npy`, `test_t.npy` - Test data

Where:
- `*_x.npy`: Feature matrices (12 features f0-f11)
- `*_y.npy`: Target labels
- `*_t.npy`: Treatment assignments

## Data Split

- **Training**: 65% of data
- **Validation**: 7% of data (20% of test portion)
- **Test**: 28% of data (80% of test portion)

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Example: Quick Experiment

For a quick experiment with 1% of the data:
```bash
python prepare_criteo_data.py --sample_ratio 0.01
```

This will process only 1% of the original dataset, making it suitable for rapid prototyping and testing. 