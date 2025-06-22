import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklift.metrics import uplift_auc_score, uplift_at_k
import argparse
import os

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_criteo_data(data_dir="data"):
    """Load Criteo data from numpy files"""
    print(f"Loading data from {data_dir}...")
    
    # Load training data
    X_train = np.load(os.path.join(data_dir, "train_x.npy"))
    y_train = np.load(os.path.join(data_dir, "train_y.npy"))
    t_train = np.load(os.path.join(data_dir, "train_t.npy"))
    
    # Load validation data
    X_val = np.load(os.path.join(data_dir, "val_x.npy"))
    y_val = np.load(os.path.join(data_dir, "val_y.npy"))
    t_val = np.load(os.path.join(data_dir, "val_t.npy"))
    
    # Load test data
    X_test = np.load(os.path.join(data_dir, "test_x.npy"))
    y_test = np.load(os.path.join(data_dir, "test_y.npy"))
    t_test = np.load(os.path.join(data_dir, "test_t.npy"))
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return (X_train, y_train, t_train), (X_val, y_val, t_val), (X_test, y_test, t_test)

def preprocess_data(X_train, y_train, t_train, X_val, y_val, t_val, X_test, y_test, t_test):
    """Preprocess the data: normalize features and convert to tensors"""
    
    # Normalize features using training data statistics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    t_train_tensor = torch.tensor(t_train, dtype=torch.float32).unsqueeze(1)
    
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    t_val_tensor = torch.tensor(t_val, dtype=torch.float32).unsqueeze(1)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    t_test_tensor = torch.tensor(t_test, dtype=torch.float32).unsqueeze(1)
    
    return (X_train_tensor, y_train_tensor, t_train_tensor), \
           (X_val_tensor, y_val_tensor, t_val_tensor), \
           (X_test_tensor, y_test_tensor, t_test_tensor)

class CausalDataset(Dataset):
    """PyTorch Dataset for causal inference data"""
    def __init__(self, X, y, t):
        self.X = X
        self.y = y
        self.t = t

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.t[idx]

class DragonNet(nn.Module):
    def __init__(self, input_dim):
        super(DragonNet, self).__init__()
        
        # Shared representation layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, 100),
            nn.ELU()
        )
        
        # Treatment prediction head (Propensity Score)
        self.propensity_head = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        
        # Outcome prediction heads
        self.outcome_head_0 = nn.Sequential(nn.Linear(100, 1))  # Y(0)
        self.outcome_head_1 = nn.Sequential(nn.Linear(100, 1))  # Y(1)

        # Learnable epsilon (initialized small)
        self.epsilon = nn.Parameter(torch.tensor(1e-6))

    def forward(self, x):
        representation = self.shared(x)
        
        # Predict treatment probability
        e_x = self.propensity_head(representation)
        
        # Predict potential outcomes
        y0 = self.outcome_head_0(representation)
        y1 = self.outcome_head_1(representation)
        
        return e_x, y0, y1

def make_regression_loss(y_0_pred, y_1_pred, y_true, t_true):
    """Compute regression loss for potential outcomes"""
    loss0 = (1 - t_true) * torch.square(y_0_pred - y_true)
    loss1 = t_true * torch.square(y_1_pred - y_true)
    loss = loss0 + loss1
    return torch.mean(loss)

def make_binary_classification_loss(t_pred, t_true):
    """Compute binary classification loss for treatment prediction"""
    return nn.BCELoss()(t_pred, t_true)

def make_targeted_regularization_loss(e_x, y0_pred, y1_pred, Y, T, epsilon):
    """Compute the doubly robust loss with targeted regularization"""
    
    # Compute predicted outcome based on treatment
    y_pred = T * y1_pred + (1 - T) * y0_pred
    
    # Compute inverse probability weights
    e_x = torch.clamp(e_x, 1e-6, 1 - 1e-6)  # Avoid division by zero
    weight = (T - e_x) / (e_x * (1 - e_x))
    
    # Compute y_pred_tilde (corrected y_pred with propensity scores)
    y_pred_tilde = y_pred + epsilon * weight
    
    # Targeted regularization loss
    t_loss = torch.mean((Y - y_pred_tilde) ** 2)
    
    return t_loss

def train_dragonnet(model, train_loader, train_for_eval_loader, val_loader, optimizer, num_epochs, alpha=0.1, beta=0.1, device=device):
    """Train the DragonNet model"""
    print("Starting training...")
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_regression_loss = 0
        total_bce_loss = 0
        total_t_loss = 0
        
        for batch_X, batch_y, batch_t in train_loader:
            batch_X, batch_y, batch_t = batch_X.to(device), batch_y.to(device), batch_t.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            e_x, y_0_pred, y_1_pred = model(batch_X)
            
            # Compute losses
            regression_loss = make_regression_loss(y_0_pred, y_1_pred, batch_y, batch_t)
            bce_loss = make_binary_classification_loss(e_x, batch_t)
            vanila_loss = regression_loss + alpha * bce_loss
            
            t_loss = make_targeted_regularization_loss(e_x, y_0_pred, y_1_pred, batch_y, batch_t, model.epsilon)
            loss = vanila_loss + beta * t_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_regression_loss += regression_loss.item()
            total_bce_loss += bce_loss.item()
            total_t_loss += t_loss.item()

        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y, batch_t in val_loader:
                batch_X, batch_y, batch_t = batch_X.to(device), batch_y.to(device), batch_t.to(device)
                
                e_x, y_0_pred, y_1_pred = model(batch_X)
                
                regression_loss = make_regression_loss(y_0_pred, y_1_pred, batch_y, batch_t)
                bce_loss = make_binary_classification_loss(e_x, batch_t)
                vanila_loss = regression_loss + alpha * bce_loss
                
                t_loss = make_targeted_regularization_loss(e_x, y_0_pred, y_1_pred, batch_y, batch_t, model.epsilon)
                loss = vanila_loss + beta * t_loss
                
                total_val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_regression_loss = total_regression_loss / len(train_loader)
        avg_bce_loss = total_bce_loss / len(train_loader)
        avg_t_loss = total_t_loss / len(train_loader)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            print(f"  Regression: {avg_regression_loss:.4f}, BCE: {avg_bce_loss:.4f}, T-Loss: {avg_t_loss:.4f}")
            print(f"  Epsilon: {model.epsilon.item():.6f}")

            # Evaluate model on train set
            print("\n" + "="*50)
            print("TRAIN SET EVALUATION")
            print("="*50)
            tau_hat_train, y_true_train, t_true_train = evaluate_dragonnet(model, train_for_eval_loader, device)

            # Evaluate model on validation set
            print("\n" + "="*50)
            print("VALIDATION SET EVALUATION")
            print("="*50)
            tau_hat_val, y_true_val, t_true_val = evaluate_dragonnet(model, val_loader, device)
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return model

def evaluate_dragonnet(model, data_loader, device=device):
    """Evaluate the DragonNet model and compute treatment effects using uplift metrics"""
    model.eval()
    tau_hat_dragonnet = []
    y_true_list = []
    t_true_list = []
    
    with torch.no_grad():
        for batch_X, batch_y, batch_t in data_loader:
            batch_X = batch_X.to(device)
            e_x_test, y0_pred_test, y1_pred_test = model(batch_X)
            
            # Compute treatment effects
            tau_batch = (y1_pred_test - y0_pred_test).cpu().numpy()
            tau_hat_dragonnet.extend(tau_batch)
            
            y_true_list.extend(batch_y.numpy())
            t_true_list.extend(batch_t.numpy())
    
    tau_hat_dragonnet = np.array(tau_hat_dragonnet).flatten()
    y_true_array = np.array(y_true_list).flatten()
    t_true_array = np.array(t_true_list).flatten()
    
    # Compute uplift metrics
    print(f"\nModel Evaluation:")
    print(f"Average predicted treatment effect: {np.mean(tau_hat_dragonnet):.4f}")
    print(f"Std of predicted treatment effects: {np.std(tau_hat_dragonnet):.4f}")
    
    # Compute AUUC (Area Under Uplift Curve)
    try:
        auuc_score = uplift_auc_score(y_true_array, tau_hat_dragonnet, t_true_array)
        print(f"AUUC Score: {auuc_score:.4f}")
    except Exception as e:
        print(f"Could not compute AUUC: {e}")
    
    # Compute Uplift at K for different K values
    k_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    for k in k_values:
        try:
            uplift_k = uplift_at_k(y_true_array, tau_hat_dragonnet, t_true_array, k=k, strategy="overall")
            print(f"Uplift at {int(k*100)}%: {uplift_k:.4f}")
        except Exception as e:
            print(f"Could not compute Uplift at {int(k*100)}%: {e}")
    
    
    return tau_hat_dragonnet, y_true_array, t_true_array

def main():
    parser = argparse.ArgumentParser(description="Train DragonNet on Criteo data")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the data files')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='Weight for BCE loss')
    parser.add_argument('--beta', type=float, default=0.1, help='Weight for targeted regularization loss')
    args = parser.parse_args()
    
    # Load data
    (X_train, y_train, t_train), (X_val, y_val, t_val), (X_test, y_test, t_test) = load_criteo_data(args.data_dir)
    
    # Preprocess data
    (X_train_tensor, y_train_tensor, t_train_tensor), \
    (X_val_tensor, y_val_tensor, t_val_tensor), \
    (X_test_tensor, y_test_tensor, t_test_tensor) = preprocess_data(
        X_train, y_train, t_train, X_val, y_val, t_val, X_test, y_test, t_test
    )
    
    # Create data loaders
    train_dataset = CausalDataset(X_train_tensor, y_train_tensor, t_train_tensor)
    val_dataset = CausalDataset(X_val_tensor, y_val_tensor, t_val_tensor)
    test_dataset = CausalDataset(X_test_tensor, y_test_tensor, t_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_for_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = DragonNet(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Model initialized with {input_dim} input features")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    model = train_dragonnet(
        model, train_loader, train_for_eval_loader, val_loader, optimizer, 
        num_epochs=args.num_epochs, 
        alpha=args.alpha, 
        beta=args.beta, 
        device=device
    )
    
    # Evaluate model on test set
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    tau_hat_test, y_true_test, t_true_test = evaluate_dragonnet(model, test_loader, device)
    
    # Evaluate model on validation set
    print("\n" + "="*50)
    print("VALIDATION SET EVALUATION")
    print("="*50)
    tau_hat_val, y_true_val, t_true_val = evaluate_dragonnet(model, val_loader, device)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()