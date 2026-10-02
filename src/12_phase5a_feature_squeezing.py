from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARRAYS_DIR = PROJECT_ROOT / "data" / "processed" / "baseline_arrays"
MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

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


def load_model(device):
    checkpoint_path = MODEL_DIR / "pytorch_clean_baseline.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_dim = int(checkpoint["input_dim"])
    model = IDSClassifier(input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, input_dim


def squeeze_features(X, bins=256):
    # X is already scaled to [0, 1].
    # Quantize to fixed precision and keep in valid range.
    Xq = np.round(X * (bins - 1)) / (bins - 1)
    return np.clip(Xq, 0.0, 1.0).astype(np.float32)


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    required = [
        ARRAYS_DIR / "X_test_scaled.npy",
        ARRAYS_DIR / "y_test.npy",
        ARRAYS_DIR / "pytorch_correct_attack_test_indices.npy",
        MODEL_DIR / "pytorch_clean_baseline.pt",
        METRICS_DIR / "feature_feasibility_table.csv",
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    X_test = np.load(ARRAYS_DIR / "X_test_scaled.npy").astype(np.float32)
    y_test = np.load(ARRAYS_DIR / "y_test.npy").astype(int)
    attack_indices = np.load(ARRAYS_DIR / "pytorch_correct_attack_test_indices.npy").astype(int)

    feasibility = pd.read_csv(METRICS_DIR / "feature_feasibility_table.csv")
    mask = feasibility["perturbation_allowed"].astype(int).to_numpy()

    model, input_dim = load_model(device)

    bins = 256
    X_test_sq = squeeze_features(X_test, bins=bins)

    print("Evaluating clean squeezed model input...")
    clean_pred, clean_score = predict(model, X_test_sq, device)
    clean_metrics = metrics(y_test, clean_pred, clean_score)

    X_attack = X_test[attack_indices]
    print("Attack cohort rows:", len(X_attack))

    eps_values = [0.005, 0.010, 0.020, 0.050, 0.100]
    rows = []

    for method in ["FGSM", "PGD"]:
        for eps in eps_values:
            if method == "FGSM":
                X_adv = targeted_fgsm(model, X_attack, eps, mask, device)
                iterations = 1
                eps_step = None
            else:
                eps_step = eps / 4.0
                X_adv = targeted_pgd(model, X_attack, eps, eps_step, 40, mask, device)
                iterations = 40

            X_adv_sq = squeeze_features(X_adv, bins=bins)

            adv_pred, adv_score = predict(model, X_adv_sq, device)

            evasion_count = int((adv_pred == 0).sum())
            evasion_rate = float(evasion_count / len(X_attack))
            post_attack_recall = float((adv_pred == 1).mean())

            row = {
                "defense": "feature_squeezing",
                "bins": bins,
                "attack_method": method,
                "constraint_mode": "constrained",
                "epsilon": eps,
                "eps_step": eps_step,
                "iterations": iterations,
                "attack_rows": int(len(X_attack)),
                "evasion_count": evasion_count,
                "evasion_rate": evasion_rate,
                "post_attack_recall": post_attack_recall,
                "avg_linf_before_squeeze": float(np.max(np.abs(X_adv - X_attack), axis=1).mean()),
                "avg_l1_before_squeeze": float(np.abs(X_adv - X_attack).sum(axis=1).mean()),
                "avg_changed_feature_count_before_squeeze": float((np.abs(X_adv - X_attack) > 1e-7).sum(axis=1).mean())
            }

            rows.append(row)

            print(
                f"{method:4s} | eps={eps:.3f} | "
                f"feature_squeezed_evasion={evasion_rate:.4f} | "
                f"recall={post_attack_recall:.4f}"
            )

    result = {
        "defense": "feature_squeezing",
        "bins": bins,
        "clean_metrics_after_squeezing": clean_metrics,
        "attack_results": rows
    }

    csv_out = METRICS_DIR / "phase5a_feature_squeezing_attack_summary.csv"
    json_out = METRICS_DIR / "phase5a_feature_squeezing_summary.json"
    complete_out = METRICS_DIR / "phase5a_feature_squeezing_complete.txt"

    pd.DataFrame(rows).to_csv(csv_out, index=False)

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    complete_out.write_text(
        "Phase 5A feature squeezing defense complete.\n",
        encoding="utf-8"
    )

    print("\nClean metrics after squeezing:")
    print(json.dumps(clean_metrics, indent=2))

    print("\nSaved:")
    print(" ", csv_out)
    print(" ", json_out)
    print(" ", complete_out)

    print("\nPhase 5A feature squeezing defense complete.")


if __name__ == "__main__":
    main()
