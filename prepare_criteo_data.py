import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Prepare Criteo Uplift Data")
    parser.add_argument('--label', type=str, default='conversion', choices=['exposure', 'visit', 'conversion'], help='Label column name')
    parser.add_argument('--treatment', type=str, default='treatment', choices=['treatment', 'exposure', 'visit'], help='Treatment column name')
    parser.add_argument('--input', type=str, default='criteo-uplift-v2.1.csv', help='Input CSV file')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory for .npy files')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='Sample ratio (0.0 to 1.0) to use from input data')
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Read the CSV file
    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input)

    # Sample data if sample_ratio is less than 1.0
    if args.sample_ratio < 1.0:
        sample_size = int(len(df) * args.sample_ratio)
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample_size} rows ({args.sample_ratio:.1%} of original data)")

    # Extract features, label, and treatment
    feature_cols = [f"f{i}" for i in range(12)]
    X = df[feature_cols].to_numpy()
    Y = df[args.label].to_numpy()
    T = df[args.treatment].to_numpy()

    # Split into train, validation, and test
    X_train, X_temp, Y_train, Y_temp, T_train, T_temp = train_test_split(
        X, Y, T, test_size=0.35, random_state=42, stratify=Y)
    X_val, X_test, Y_val, Y_test, T_val, T_test = train_test_split(
        X_temp, Y_temp, T_temp, test_size=0.2/0.35, random_state=42, stratify=Y_temp)

    # Save splits
    np.save(f"{args.output_dir}/train_x.npy", X_train)
    np.save(f"{args.output_dir}/train_y.npy", Y_train)
    np.save(f"{args.output_dir}/train_t.npy", T_train)
    np.save(f"{args.output_dir}/val_x.npy", X_val)
    np.save(f"{args.output_dir}/val_y.npy", Y_val)
    np.save(f"{args.output_dir}/val_t.npy", T_val)
    np.save(f"{args.output_dir}/test_x.npy", X_test)
    np.save(f"{args.output_dir}/test_y.npy", Y_test)
    np.save(f"{args.output_dir}/test_t.npy", T_test)
    print(f"Saved splits to {args.output_dir}/")

if __name__ == "__main__":
    main() 