import numpy as np
import argparse
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklift.metrics import uplift_auc_score, uplift_at_k
import sklearn

def dr_learner(
    X: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    prop_model=None,
    outcome_model=None,
    final_model=None,
    n_splits: int = 2
):
    """
    Two-fold Doubly-Robust Learner.
    Returns a fitted final_model that predicts tau(x).
    """
    # Fix for scikit-learn 1.6.1 bug: explicitly create models instead of using 'or'
    if prop_model is None:
        prop_model = LogisticRegression(max_iter=1000)
    if outcome_model is None:
        outcome_model = RandomForestRegressor(n_estimators=100, random_state=42)
    if final_model is None:
        final_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # placeholder for pseudo-outcome
    phi = np.zeros_like(y, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        # split for nuisance estimation vs pseudo-outcome
        X_train, A_train, y_train = X[train_idx], A[train_idx], y[train_idx]
        X_test,  A_test,  y_test  = X[test_idx],  A[test_idx],  y[test_idx]

        # 1) propensity score p(x)
        try:
            pm = clone(prop_model).fit(X_train, A_train)
            pi_hat = pm.predict_proba(X_test)[:, 1]
        except Exception as e:
            print(f"Error in propensity score estimation: {e}")
            raise

        # 2) outcome models mu0(x), mu1(x)
        try:
            # Check if we have enough samples for each treatment group
            n_control = np.sum(A_train == 0)
            n_treated = np.sum(A_train == 1)
            
            if n_control == 0 or n_treated == 0:
                raise ValueError(f"No samples in one of the treatment groups (control: {n_control}, treated: {n_treated})")
            
            m0 = clone(outcome_model).fit(X_train[A_train == 0], y_train[A_train == 0])
            m1 = clone(outcome_model).fit(X_train[A_train == 1], y_train[A_train == 1])
            mu0 = m0.predict(X_test)
            mu1 = m1.predict(X_test)
        except Exception as e:
            print(f"Error in outcome model estimation: {e}")
            raise

        # 3) compute pseudo-outcome
        phi[test_idx] = (
            (A_test - pi_hat) / (pi_hat * (1 - pi_hat)) * 
            (y_test - np.where(A_test == 1, mu1, mu0))
            + (mu1 - mu0)
        )

    # 4) final regression of phi on X
    try:
        final_model.fit(X, phi)
    except Exception as e:
        print(f"Error in final model fitting: {e}")
        raise
        
    return final_model

def main():
    parser = argparse.ArgumentParser("Run DR-learner uplift modeling")
    parser.add_argument('--data_folder', type=str, default='data', help='Data folder path')
    args = parser.parse_args()

    # --- load data ---
    X_train = np.load(f"{args.data_folder}/train_x.npy")
    y_train = np.load(f"{args.data_folder}/train_y.npy").flatten()
    t_train = np.load(f"{args.data_folder}/train_t.npy").flatten()

    X_val   = np.load(f"{args.data_folder}/val_x.npy")
    y_val   = np.load(f"{args.data_folder}/val_y.npy").flatten()
    t_val   = np.load(f"{args.data_folder}/val_t.npy").flatten()

    X_test  = np.load(f"{args.data_folder}/test_x.npy")
    y_test  = np.load(f"{args.data_folder}/test_y.npy").flatten()
    t_test  = np.load(f"{args.data_folder}/test_t.npy").flatten()

    # optional sanity checks
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}, t={t_train.shape}")
    print(f"Treatment rate: {t_train.mean():.3f}, Outcome rate: {y_train.mean():.3f}")

    # --- fit DR-learner ---
    print("Fitting DR-learner...")
    model = dr_learner(
        X_train, y_train, t_train,
        prop_model=LogisticRegression(max_iter=1000),
        outcome_model=RandomForestRegressor(n_estimators=100),
        final_model=RandomForestRegressor(n_estimators=100),
        n_splits=2
    )

    # --- predict uplift ---
    tau_val  = model.predict(X_val)
    tau_test = model.predict(X_test)

    # --- compute metrics via scikit-uplift ---
    val_auuc  = uplift_auc_score(y_true=y_val,   uplift=tau_val,   treatment=t_val)
    test_auuc = uplift_auc_score(y_true=y_test,  uplift=tau_test,  treatment=t_test)

    print("\nValidation Metrics (DR-learner):")
    print(f"  AUUC: {val_auuc:.4f}")
    for k in [0.05, 0.10, 0.25, 0.50]:
        u = uplift_at_k(y_true=y_val, uplift=tau_val, treatment=t_val,
                        strategy='overall', k=k)
        print(f"  Uplift@{int(k*100)}%: {u:.4f}")

    print("\nTest Metrics (DR-learner):")
    print(f"  AUUC: {test_auuc:.4f}")
    for k in [0.05, 0.10, 0.25, 0.50]:
        u = uplift_at_k(y_true=y_test, uplift=tau_test, treatment=t_test,
                        strategy='overall', k=k)
        print(f"  Uplift@{int(k*100)}%: {u:.4f}")

if __name__ == "__main__":
    main()

