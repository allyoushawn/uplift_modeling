import numpy as np
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from causalml.inference.meta import BaseTRegressor
from sklift.metrics import uplift_auc_score, uplift_at_k
from typing import Dict

def main():
    parser = argparse.ArgumentParser(description="Run T-learner uplift modeling")
    parser.add_argument('--data_folder', type=str, default='data', help='Folder containing the data files')
    args = parser.parse_args()
    
    print('Running T-learner...')
    
    # Paths using the data_folder argument
    X_train = np.load(f"{args.data_folder}/train_x.npy")
    y_train = np.load(f"{args.data_folder}/train_y.npy")
    t_train = np.load(f"{args.data_folder}/train_t.npy")

    X_val = np.load(f"{args.data_folder}/val_x.npy")
    y_val = np.load(f"{args.data_folder}/val_y.npy")
    t_val = np.load(f"{args.data_folder}/val_t.npy")

    X_test = np.load(f"{args.data_folder}/test_x.npy")
    y_test = np.load(f"{args.data_folder}/test_y.npy")
    t_test = np.load(f"{args.data_folder}/test_t.npy")

    # Debug: Check data characteristics
    print(f"Training data - X shape: {X_train.shape}, y shape: {y_train.shape}, t shape: {t_train.shape}")
    print(f"Training data - y unique: {np.unique(y_train)}, t unique: {np.unique(t_train)}")
    print(f"Training data - y mean: {y_train.mean():.6f}, t mean: {t_train.mean():.6f}")
    print(f"Training data - treatment balance: {np.sum(t_train == 1)}/{len(t_train)} = {t_train.mean():.3f}")
    
    # Check for potential issues
    if y_train.mean() < 0.01:
        print("WARNING: Very low conversion rate in training data!")
    if t_train.mean() < 0.1 or t_train.mean() > 0.9:
        print("WARNING: Treatment imbalance detected!")
    
    # Initialize a T-learner with logistic regression as the base learner
    t_learner = BaseTRegressor(
        learner=LogisticRegression(max_iter=1000),
        control_name=0
    )

    # Fit on training data
    t_learner.fit(X=X_train, treatment=t_train, y=y_train)

    # --- Validation set evaluation ---
    tau_val = t_learner.predict(X_val)  # individual treatment effect estimates
    
    # Ensure arrays are 1-dimensional
    y_val_1d = y_val.flatten() if y_val.ndim > 1 else y_val
    t_val_1d = t_val.flatten() if t_val.ndim > 1 else t_val
    tau_val_1d = tau_val.flatten() if tau_val.ndim > 1 else tau_val
    
    
    # Calculate metrics using sklift implementation
    val_auuc = uplift_auc_score(y_val_1d, tau_val_1d, t_val_1d)
    val_uplift_5 = uplift_at_k(y_val_1d, tau_val_1d, t_val_1d, strategy='overall', k=0.05)
    val_uplift_10 = uplift_at_k(y_val_1d, tau_val_1d, t_val_1d, strategy='overall', k=0.10)
    val_uplift_25 = uplift_at_k(y_val_1d, tau_val_1d, t_val_1d, strategy='overall', k=0.25)
    val_uplift_50 = uplift_at_k(y_val_1d, tau_val_1d, t_val_1d, strategy='overall', k=0.50)

    print("Validation Metrics (sklift):")
    print(f"  AUUC: {val_auuc:.4f}")
    print(f"  Uplift@5%: {val_uplift_5:.4f}")
    print(f"  Uplift@10%: {val_uplift_10:.4f}")
    print(f"  Uplift@25%: {val_uplift_25:.4f}")
    print(f"  Uplift@50%: {val_uplift_50:.4f}")
    
    print()

    # --- Test set evaluation ---
    tau_test = t_learner.predict(X_test)
    
    # Ensure arrays are 1-dimensional
    y_test_1d = y_test.flatten() if y_test.ndim > 1 else y_test
    t_test_1d = t_test.flatten() if t_test.ndim > 1 else t_test
    tau_test_1d = tau_test.flatten() if tau_test.ndim > 1 else tau_test
    
    
    # Calculate metrics using sklift implementation
    test_auuc = uplift_auc_score(y_test_1d, tau_test_1d, t_test_1d)
    test_uplift_5 = uplift_at_k(y_test_1d, tau_test_1d, t_test_1d, strategy='overall', k=0.05)
    test_uplift_10 = uplift_at_k(y_test_1d, tau_test_1d, t_test_1d, strategy='overall', k=0.10)
    test_uplift_25 = uplift_at_k(y_test_1d, tau_test_1d, t_test_1d, strategy='overall', k=0.25)
    test_uplift_50 = uplift_at_k(y_test_1d, tau_test_1d, t_test_1d, strategy='overall', k=0.50)

    print("Test Metrics (sklift):")
    print(f"  AUUC: {test_auuc:.4f}")
    print(f"  Uplift@5%: {test_uplift_5:.4f}")
    print(f"  Uplift@10%: {test_uplift_10:.4f}")
    print(f"  Uplift@25%: {test_uplift_25:.4f}")
    print(f"  Uplift@50%: {test_uplift_50:.4f}")

if __name__ == "__main__":
    main()
