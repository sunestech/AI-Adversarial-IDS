from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARRAYS_DIR = PROJECT_ROOT / "data" / "processed" / "baseline_arrays"
BASELINE_MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
DEFENSE_MODEL_DIR = PROJECT_ROOT / "models" / "defense"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

DEFENSE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


class IDSClassifier(nn.Module):
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
        return self.net(x.float())


def predict(model, X, device, batch_size=4096):
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


def metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    attack_precision = precision[1]
    attack_recall = recall[1]

    if attack_precision + attack_recall == 0:
        attack_f2 = 0.0
    else:
        attack_f2 = (5 * attack_precision * attack_recall) / ((4 * attack_precision) + attack_recall)

    return {
        "accuracy": float(acc),
        "benign_precision": float(precision[0]),
        "benign_recall": float(recall[0]),
        "benign_f1": float(f1[0]),
        "benign_support": int(support[0]),
        "attack_precision": float(precision[1]),
        "attack_recall": float(recall[1]),
        "attack_f1": float(f1[1]),
        "attack_f2": float(attack_f2),
        "attack_support": int(support[1]),
        "false_positive_rate": float(fpr),
        "attack_pr_auc": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": cm.tolist()
    }


def targeted_pgd(model, X, eps, eps_step, max_iter, mask, device):
    x_orig = torch.tensor(X, dtype=torch.float32, device=device)
    x_adv = x_orig.clone().detach()

    target = torch.zeros(x_adv.shape[0], dtype=torch.long, device=device)
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(1, -1)

    for _ in range(max_iter):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = nn.CrossEntropyLoss()(logits, target)

        model.zero_grad()
        loss.backward()

        grad_sign = x_adv.grad.detach().sign()

        x_adv = x_adv - eps_step * grad_sign * mask_tensor
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    return x_adv.cpu().numpy().astype(np.float32)


def targeted_fgsm(model, X, eps, mask, device):
    x_orig = torch.tensor(X, dtype=torch.float32, device=device)
    x_adv = x_orig.clone().detach().requires_grad_(True)

    target = torch.zeros(x_adv.shape[0], dtype=torch.long, device=device)
    logits = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, target)

    model.zero_grad()
    loss.backward()

    grad_sign = x_adv.grad.detach().sign()
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(1, -1)

    x_adv = x_adv - eps * grad_sign * mask_tensor
    x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach().cpu().numpy().astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    required = [
        ARRAYS_DIR / "X_train_scaled.npy",
        ARRAYS_DIR / "X_val_scaled.npy",
        ARRAYS_DIR / "X_test_scaled.npy",
        ARRAYS_DIR / "y_train.npy",
        ARRAYS_DIR / "y_val.npy",
        ARRAYS_DIR / "y_test.npy",
        ARRAYS_DIR / "pytorch_correct_attack_test_indices.npy",
        BASELINE_MODEL_DIR / "pytorch_clean_baseline.pt",
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

    attack_indices = np.load(ARRAYS_DIR / "pytorch_correct_attack_test_indices.npy").astype(int)

    feasibility = pd.read_csv(METRICS_DIR / "feature_feasibility_table.csv")
    mask = feasibility["perturbation_allowed"].astype(int).to_numpy()

    baseline_checkpoint = torch.load(BASELINE_MODEL_DIR / "pytorch_clean_baseline.pt", map_location=device)
    input_dim = int(baseline_checkpoint["input_dim"])

    baseline_model = IDSClassifier(input_dim=input_dim).to(device)
    baseline_model.load_state_dict(baseline_checkpoint["model_state_dict"])
    baseline_model.eval()

    rng = np.random.default_rng(2026)

    attack_train_indices = np.where(y_train == 1)[0]
    adv_train_count = min(30000, len(attack_train_indices))
    adv_idx = rng.choice(attack_train_indices, size=adv_train_count, replace=False)

    print("Creating adversarial training examples from attack rows:", adv_train_count)

    X_adv_train = targeted_pgd(
        model=baseline_model,
        X=X_train[adv_idx],
        eps=0.05,
        eps_step=0.0125,
        max_iter=10,
        mask=mask,
        device=device
    )

    y_adv_train = np.ones(len(X_adv_train), dtype=int)

    X_aug = np.vstack([X_train, X_adv_train]).astype(np.float32)
    y_aug = np.concatenate([y_train, y_adv_train]).astype(int)

    shuffle_idx = rng.permutation(len(X_aug))
    X_aug = X_aug[shuffle_idx]
    y_aug = y_aug[shuffle_idx]

    print("Clean train rows:", len(X_train))
    print("Adversarial train rows added:", len(X_adv_train))
    print("Augmented train rows:", len(X_aug))

    defense_model = IDSClassifier(input_dim=input_dim).to(device)

    class_counts = np.bincount(y_aug, minlength=2)
    class_weights = class_counts.sum() / (2.0 * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(defense_model.parameters(), lr=0.001)

    train_ds = TensorDataset(
        torch.tensor(X_aug, dtype=torch.float32),
        torch.tensor(y_aug, dtype=torch.long)
    )

    loader = DataLoader(train_ds, batch_size=2048, shuffle=True)

    best_state = None
    best_val_attack_f1 = -1.0

    print("\nTraining adversarially trained defense model...")

    for epoch in range(1, 16):
        defense_model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = defense_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_ds)

        val_pred, val_score = predict(defense_model, X_val, device)
        val_metrics = metrics(y_val, val_pred, val_score)

        print(
            f"Epoch {epoch:02d} | loss={avg_loss:.6f} | "
            f"val_acc={val_metrics['accuracy']:.6f} | "
            f"val_attack_recall={val_metrics['attack_recall']:.6f} | "
            f"val_attack_f1={val_metrics['attack_f1']:.6f}"
        )

        if val_metrics["attack_f1"] > best_val_attack_f1:
            best_val_attack_f1 = val_metrics["attack_f1"]
            best_state = {
                "model_state_dict": defense_model.state_dict(),
                "input_dim": input_dim,
                "adversarial_training_eps": 0.05,
                "adversarial_training_rows": int(len(X_adv_train)),
                "random_seed": 2026,
                "best_val_attack_f1": float(best_val_attack_f1),
            }

    defense_model.load_state_dict(best_state["model_state_dict"])
    defense_model.eval()

    torch.save(best_state, DEFENSE_MODEL_DIR / "pytorch_adversarial_training_defense.pt")

    print("\nEvaluating clean test performance...")
    clean_pred, clean_score = predict(defense_model, X_test, device)
    clean_metrics = metrics(y_test, clean_pred, clean_score)

    X_attack = X_test[attack_indices]
    eps_values = [0.005, 0.010, 0.020, 0.050, 0.100]

    rows = []

    for method in ["FGSM", "PGD"]:
        for eps in eps_values:
            if method == "FGSM":
                X_adv = targeted_fgsm(defense_model, X_attack, eps, mask, device)
                iterations = 1
                eps_step = None
            else:
                eps_step = eps / 4.0
                X_adv = targeted_pgd(defense_model, X_attack, eps, eps_step, 40, mask, device)
                iterations = 40

            adv_pred, adv_score = predict(defense_model, X_adv, device)

            evasion_count = int((adv_pred == 0).sum())
            evasion_rate = float(evasion_count / len(X_attack))
            post_attack_recall = float((adv_pred == 1).mean())

            rows.append({
                "defense": "adversarial_training",
                "attack_method": method,
                "constraint_mode": "constrained",
                "epsilon": eps,
                "eps_step": eps_step,
                "iterations": iterations,
                "attack_rows": int(len(X_attack)),
                "evasion_count": evasion_count,
                "evasion_rate": evasion_rate,
                "post_attack_recall": post_attack_recall,
                "avg_linf": float(np.max(np.abs(X_adv - X_attack), axis=1).mean()),
                "avg_l1": float(np.abs(X_adv - X_attack).sum(axis=1).mean()),
                "avg_changed_feature_count": float((np.abs(X_adv - X_attack) > 1e-7).sum(axis=1).mean())
            })

            print(
                f"{method:4s} | eps={eps:.3f} | "
                f"adversarial_training_evasion={evasion_rate:.4f} | "
                f"recall={post_attack_recall:.4f}"
            )

    result = {
        "defense": "adversarial_training",
        "clean_metrics_after_defense": clean_metrics,
        "training_details": {
            "adversarial_training_rows": int(len(X_adv_train)),
            "adversarial_training_eps": 0.05,
            "adversarial_training_attack": "constrained PGD",
            "epochs": 15
        },
        "attack_results": rows
    }

    csv_out = METRICS_DIR / "phase5b_adversarial_training_attack_summary.csv"
    json_out = METRICS_DIR / "phase5b_adversarial_training_summary.json"
    complete_out = METRICS_DIR / "phase5b_adversarial_training_complete.txt"

    pd.DataFrame(rows).to_csv(csv_out, index=False)

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    complete_out.write_text(
        "Phase 5B adversarial training defense complete.\n",
        encoding="utf-8"
    )

    print("\nClean metrics after adversarial training:")
    print(json.dumps(clean_metrics, indent=2))

    print("\nSaved:")
    print(" ", DEFENSE_MODEL_DIR / "pytorch_adversarial_training_defense.pt")
    print(" ", csv_out)
    print(" ", json_out)
    print(" ", complete_out)

    print("\nPhase 5B adversarial training defense complete.")


if __name__ == "__main__":
    main()
