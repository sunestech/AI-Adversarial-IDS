from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARRAYS_DIR = PROCESSED_DIR / "baseline_arrays"
MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
SURROGATE_DIR = PROJECT_ROOT / "models" / "surrogate"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

SURROGATE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


class TargetIDSClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


class SurrogateIDSClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LeakyReLU(0.10),
            nn.Dropout(0.20),
            nn.Linear(96, 48),
            nn.LeakyReLU(0.10),
            nn.Linear(48, 24),
            nn.LeakyReLU(0.10),
            nn.Linear(24, 2)
        )

    def forward(self, x):
        return self.net(x)


def load_target_pytorch(device):
    checkpoint_path = MODEL_DIR / "pytorch_clean_baseline.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_dim = int(checkpoint["input_dim"])
    model = TargetIDSClassifier(input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, input_dim


def train_surrogate(X_train, y_train, X_val, y_val, device, smoke=False):
    input_dim = X_train.shape[1]
    model = SurrogateIDSClassifier(input_dim=input_dim).to(device)

    rng = np.random.default_rng(2026)

    train_size = len(X_train)
    if smoke:
        subset_size = min(25000, train_size)
    else:
        subset_size = train_size // 2

    subset_idx = rng.choice(train_size, size=subset_size, replace=False)

    X_sub = X_train[subset_idx]
    y_sub = y_train[subset_idx]

    class_counts = np.bincount(y_sub, minlength=2)
    class_weights = class_counts.sum() / (2.0 * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_ds = TensorDataset(
        torch.tensor(X_sub, dtype=torch.float32),
        torch.tensor(y_sub, dtype=torch.long)
    )

    loader = DataLoader(train_ds, batch_size=2048, shuffle=True)

    if smoke:
        epochs = 5
    else:
        epochs = 15

    print("Training surrogate model...")
    print("Surrogate training rows:", len(X_sub))
    print("Epochs:", epochs)

    best_state = None
    best_val_recall = -1.0

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()

        attack_mask = y_val == 1
        val_attack_recall = float((val_pred[attack_mask] == 1).mean())

        print(
            f"Epoch {epoch:02d} | loss={avg_loss:.6f} | "
            f"surrogate_val_attack_recall={val_attack_recall:.6f}"
        )

        if val_attack_recall > best_val_recall:
            best_val_recall = val_attack_recall
            best_state = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "best_val_attack_recall": best_val_recall,
                "surrogate_training_rows": int(len(X_sub)),
                "smoke": bool(smoke),
            }

    model.load_state_dict(best_state["model_state_dict"])

    torch.save(best_state, SURROGATE_DIR / "surrogate_transfer_model.pt")

    return model


def predict_torch(model, X, device, batch_size=4096):
    preds = []
    scores = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds.append(torch.argmax(probs, dim=1).cpu().numpy())
            scores.append(probs[:, 1].cpu().numpy())

    return np.concatenate(preds), np.concatenate(scores)


def targeted_pgd_surrogate(model, X, eps, eps_step, max_iter, mask, device):
    x_orig = torch.tensor(X, dtype=torch.float32).to(device)
    x_adv = x_orig.clone().detach()

    mask_tensor = torch.tensor(mask, dtype=torch.float32).to(device).view(1, -1)
    target = torch.zeros(x_adv.shape[0], dtype=torch.long).to(device)

    for _ in range(max_iter):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = nn.CrossEntropyLoss()(logits, target)

        model.zero_grad()
        loss.backward()

        grad_sign = x_adv.grad.detach().sign()

        # Targeted movement toward BENIGN class.
        x_adv = x_adv - eps_step * grad_sign * mask_tensor

        # Project to epsilon ball around original point.
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)

        # Keep scaled feature space valid.
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    return x_adv.cpu().numpy()


def validate_constraints(X_orig, X_adv, eps, mask):
    diff = np.abs(X_adv - X_orig)
    changed = diff > 1e-7

    frozen_changed = changed[:, mask == 0].sum()
    eps_violations = (diff > eps + 1e-6).sum()
    lower_violations = (X_adv < -1e-7).sum()
    upper_violations = (X_adv > 1.0 + 1e-7).sum()

    return {
        "frozen_feature_change_count": int(frozen_changed),
        "epsilon_violation_count": int(eps_violations),
        "lower_bound_violation_count": int(lower_violations),
        "upper_bound_violation_count": int(upper_violations),
        "passed": bool(
            frozen_changed == 0 and
            eps_violations == 0 and
            lower_violations == 0 and
            upper_violations == 0
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    required = [
        ARRAYS_DIR / "X_train_scaled.npy",
        ARRAYS_DIR / "X_val_scaled.npy",
        ARRAYS_DIR / "X_test_scaled.npy",
        ARRAYS_DIR / "y_train.npy",
        ARRAYS_DIR / "y_val.npy",
        ARRAYS_DIR / "y_test.npy",
        ARRAYS_DIR / "shared_correct_attack_test_indices.npy",
        MODEL_DIR / "pytorch_clean_baseline.pt",
        MODEL_DIR / "xgboost_clean_baseline.json",
        METRICS_DIR / "feature_feasibility_table.csv",
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    X_train = np.load(ARRAYS_DIR / "X_train_scaled.npy").astype(np.float32)
    X_val = np.load(ARRAYS_DIR / "X_val_scaled.npy").astype(np.float32)
    X_test = np.load(ARRAYS_DIR / "X_test_scaled.npy").astype(np.float32)

    y_train = np.load(ARRAYS_DIR / "y_train.npy").astype(int)
    y_val = np.load(ARRAYS_DIR / "y_val.npy").astype(int)
    y_test = np.load(ARRAYS_DIR / "y_test.npy").astype(int)

    shared_indices = np.load(ARRAYS_DIR / "shared_correct_attack_test_indices.npy").astype(int)

    feasibility = pd.read_csv(METRICS_DIR / "feature_feasibility_table.csv")
    mask = feasibility["perturbation_allowed"].astype(int).to_numpy()

    if len(mask) != X_test.shape[1]:
        raise ValueError("Feasibility mask length does not match feature count.")

    target_pytorch, input_dim = load_target_pytorch(device)

    xgb_model = XGBClassifier()
    xgb_model.load_model(MODEL_DIR / "xgboost_clean_baseline.json")

    surrogate = train_surrogate(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        device=device,
        smoke=args.smoke
    )

    if args.smoke:
        run_name = "phase4a_transfer_smoke"
        selected_indices = shared_indices[:512]
        eps_values = [0.01, 0.05]
        pgd_iter = 8
    else:
        run_name = "phase4a_transfer_full"
        selected_indices = shared_indices
        eps_values = [0.005, 0.010, 0.020, 0.050, 0.100]
        pgd_iter = 40

    X_attack = X_test[selected_indices]
    y_attack = y_test[selected_indices]

    print("\nRun name:", run_name)
    print("Shared target-correct attack rows:", len(X_attack))
    print("Epsilon values:", eps_values)
    print("PGD iterations:", pgd_iter)

    surrogate_base_pred, _ = predict_torch(surrogate, X_attack, device)
    pytorch_base_pred, _ = predict_torch(target_pytorch, X_attack, device)
    xgb_base_pred = xgb_model.predict(X_attack)

    rows = []

    for eps in eps_values:
        eps_step = eps / 4.0

        X_adv = targeted_pgd_surrogate(
            model=surrogate,
            X=X_attack,
            eps=eps,
            eps_step=eps_step,
            max_iter=pgd_iter,
            mask=mask,
            device=device
        )

        surrogate_adv_pred, _ = predict_torch(surrogate, X_adv, device)
        pytorch_adv_pred, _ = predict_torch(target_pytorch, X_adv, device)
        xgb_adv_pred = xgb_model.predict(X_adv)

        surrogate_evasion = float((surrogate_adv_pred == 0).mean())
        pytorch_transfer_evasion = float((pytorch_adv_pred == 0).mean())
        xgb_transfer_evasion = float((xgb_adv_pred == 0).mean())

        validation = validate_constraints(X_attack, X_adv, eps, mask)

        row = {
            "run_name": run_name,
            "attack_method": "Surrogate_PGD",
            "constraint_mode": "constrained",
            "epsilon": float(eps),
            "eps_step": float(eps_step),
            "iterations": int(pgd_iter),
            "attack_rows": int(len(X_attack)),
            "surrogate_baseline_attack_recall": float((surrogate_base_pred == 1).mean()),
            "pytorch_baseline_attack_recall": float((pytorch_base_pred == 1).mean()),
            "xgboost_baseline_attack_recall": float((xgb_base_pred == 1).mean()),
            "surrogate_whitebox_evasion_rate": surrogate_evasion,
            "pytorch_transfer_evasion_rate": pytorch_transfer_evasion,
            "xgboost_transfer_evasion_rate": xgb_transfer_evasion,
            "pytorch_post_transfer_recall": float((pytorch_adv_pred == 1).mean()),
            "xgboost_post_transfer_recall": float((xgb_adv_pred == 1).mean()),
            "avg_linf": float(np.max(np.abs(X_adv - X_attack), axis=1).mean()),
            "avg_l1": float(np.abs(X_adv - X_attack).sum(axis=1).mean()),
            "avg_changed_feature_count": float((np.abs(X_adv - X_attack) > 1e-7).sum(axis=1).mean()),
            "constraint_passed": validation["passed"],
            "frozen_feature_change_count": validation["frozen_feature_change_count"],
            "epsilon_violation_count": validation["epsilon_violation_count"],
            "lower_bound_violation_count": validation["lower_bound_violation_count"],
            "upper_bound_violation_count": validation["upper_bound_violation_count"],
        }

        rows.append(row)

        print(
            f"eps={eps:.3f} | "
            f"surrogate_evasion={surrogate_evasion:.4f} | "
            f"pytorch_transfer={pytorch_transfer_evasion:.4f} | "
            f"xgboost_transfer={xgb_transfer_evasion:.4f} | "
            f"constraints={validation['passed']}"
        )

    results_df = pd.DataFrame(rows)

    csv_out = METRICS_DIR / f"{run_name}_summary.csv"
    json_out = METRICS_DIR / f"{run_name}_summary.json"
    complete_out = METRICS_DIR / f"{run_name}_complete.txt"

    results_df.to_csv(csv_out, index=False)

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    complete_out.write_text(
        f"{run_name} complete.\n",
        encoding="utf-8"
    )

    print("\nSaved:")
    print(" ", csv_out)
    print(" ", json_out)
    print(" ", complete_out)

    print("\nPhase 4A transfer attack complete.")


if __name__ == "__main__":
    main()
