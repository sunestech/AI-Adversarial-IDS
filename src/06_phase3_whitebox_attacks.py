from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARRAYS_DIR = PROCESSED_DIR / "baseline_arrays"
MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PRED_DIR = PROJECT_ROOT / "results" / "predictions"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


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
        return self.net(x)


def load_model(device):
    checkpoint_path = MODEL_DIR / "pytorch_clean_baseline.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing PyTorch baseline: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    input_dim = int(checkpoint["input_dim"])

    model = IDSClassifier(input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, input_dim


def targeted_fgsm(model, x, eps, mask, device):
    x_orig = torch.tensor(x, dtype=torch.float32, device=device)
    x_adv = x_orig.clone().detach().requires_grad_(True)

    target = torch.zeros(x_adv.shape[0], dtype=torch.long, device=device)

    logits = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, target)

    model.zero_grad()
    loss.backward()

    # Targeted attack toward BENIGN class minimizes benign-target loss.
    grad_sign = x_adv.grad.detach().sign()
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(1, -1)

    x_adv = x_adv - eps * grad_sign * mask_tensor
    x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach().cpu().numpy()


def targeted_pgd(model, x, eps, eps_step, max_iter, mask, device):
    x_orig = torch.tensor(x, dtype=torch.float32, device=device)
    x_adv = x_orig.clone().detach()

    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(1, -1)
    target = torch.zeros(x_adv.shape[0], dtype=torch.long, device=device)

    for _ in range(max_iter):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = nn.CrossEntropyLoss()(logits, target)

        model.zero_grad()
        loss.backward()

        grad_sign = x_adv.grad.detach().sign()

        # Targeted PGD step toward BENIGN class.
        x_adv = x_adv - eps_step * grad_sign * mask_tensor

        # Project back to L-infinity epsilon ball.
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)

        # Keep scaled feature space valid.
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    return x_adv.cpu().numpy()


def predict_batches(model, x, device, batch_size=4096):
    preds = []
    scores = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds.append(torch.argmax(probs, dim=1).cpu().numpy())
            scores.append(probs[:, 1].cpu().numpy())

    return np.concatenate(preds), np.concatenate(scores)


def validate_constraints(x_orig, x_adv, eps, mask):
    diff = np.abs(x_adv - x_orig)
    changed = diff > 1e-7

    frozen_changed = changed[:, mask == 0].sum() if (mask == 0).any() else 0
    eps_violations = (diff > eps + 1e-6).sum()
    lower_violations = (x_adv < -1e-7).sum()
    upper_violations = (x_adv > 1.0 + 1e-7).sum()

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

    model, input_dim = load_model(device)

    x_test_path = ARRAYS_DIR / "X_test_scaled.npy"
    y_test_path = ARRAYS_DIR / "y_test.npy"
    indices_path = ARRAYS_DIR / "pytorch_correct_attack_test_indices.npy"
    feasibility_path = METRICS_DIR / "feature_feasibility_table.csv"

    for path in [x_test_path, y_test_path, indices_path, feasibility_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    X_test = np.load(x_test_path).astype(np.float32)
    y_test = np.load(y_test_path).astype(int)
    correct_attack_indices = np.load(indices_path).astype(int)

    feasibility = pd.read_csv(feasibility_path)
    mask = feasibility["perturbation_allowed"].astype(int).to_numpy()

    if len(mask) != X_test.shape[1]:
        raise ValueError(f"Mask length {len(mask)} does not match X_test features {X_test.shape[1]}")

    if args.smoke:
        run_name = "phase3_whitebox_smoke"
        selected_indices = correct_attack_indices[:512]
        eps_values = [0.01, 0.05]
        pgd_iter = 8
    else:
        run_name = "phase3_whitebox_full"
        selected_indices = correct_attack_indices
        eps_values = [0.005, 0.010, 0.020, 0.050, 0.100]
        pgd_iter = 40

    X_attack = X_test[selected_indices]
    y_attack = y_test[selected_indices]

    print("Run name:", run_name)
    print("Attack cohort rows:", len(selected_indices))
    print("Epsilon values:", eps_values)
    print("PGD iterations:", pgd_iter)

    base_pred, base_scores = predict_batches(model, X_attack, device)
    base_attack_detected = int((base_pred == 1).sum())

    print("Baseline detected attacks in selected cohort:", base_attack_detected)

    rows = []

    for attack_method in ["FGSM", "PGD"]:
        for constraint_mode in ["unconstrained", "constrained"]:
            for eps in eps_values:
                if constraint_mode == "unconstrained":
                    attack_mask = np.ones(X_attack.shape[1], dtype=int)
                else:
                    attack_mask = mask.copy()

                if attack_method == "FGSM":
                    X_adv = targeted_fgsm(
                        model=model,
                        x=X_attack,
                        eps=eps,
                        mask=attack_mask,
                        device=device
                    )
                    attack_iterations = 1
                else:
                    eps_step = eps / 4.0
                    X_adv = targeted_pgd(
                        model=model,
                        x=X_attack,
                        eps=eps,
                        eps_step=eps_step,
                        max_iter=pgd_iter,
                        mask=attack_mask,
                        device=device
                    )
                    attack_iterations = pgd_iter

                adv_pred, adv_scores = predict_batches(model, X_adv, device)

                evaded = adv_pred == 0
                evasion_count = int(evaded.sum())
                evasion_rate = evasion_count / len(X_attack)

                post_attack_recall = float((adv_pred == 1).sum() / len(X_attack))

                avg_linf = float(np.max(np.abs(X_adv - X_attack), axis=1).mean())
                avg_l1 = float(np.abs(X_adv - X_attack).sum(axis=1).mean())
                changed_feature_count = float(((np.abs(X_adv - X_attack) > 1e-7).sum(axis=1)).mean())

                validation = validate_constraints(X_attack, X_adv, eps, attack_mask)

                row = {
                    "run_name": run_name,
                    "attack_method": attack_method,
                    "constraint_mode": constraint_mode,
                    "epsilon": eps,
                    "eps_step": None if attack_method == "FGSM" else eps / 4.0,
                    "iterations": attack_iterations,
                    "attack_rows": int(len(X_attack)),
                    "baseline_detected_attack_rows": base_attack_detected,
                    "evasion_count": evasion_count,
                    "evasion_rate": float(evasion_rate),
                    "post_attack_recall": post_attack_recall,
                    "avg_linf": avg_linf,
                    "avg_l1": avg_l1,
                    "avg_changed_feature_count": changed_feature_count,
                    "constraint_passed": validation["passed"],
                    "frozen_feature_change_count": validation["frozen_feature_change_count"],
                    "epsilon_violation_count": validation["epsilon_violation_count"],
                    "lower_bound_violation_count": validation["lower_bound_violation_count"],
                    "upper_bound_violation_count": validation["upper_bound_violation_count"],
                }

                rows.append(row)

                print(
                    f"{attack_method:4s} | {constraint_mode:13s} | eps={eps:.3f} | "
                    f"evasion={evasion_rate:.4f} | recall={post_attack_recall:.4f} | "
                    f"constraints={validation['passed']}"
                )

    results_df = pd.DataFrame(rows)

    results_csv = METRICS_DIR / f"{run_name}_summary.csv"
    results_json = METRICS_DIR / f"{run_name}_summary.json"
    complete_txt = METRICS_DIR / f"{run_name}_complete.txt"

    results_df.to_csv(results_csv, index=False)

    with results_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    complete_txt.write_text(
        f"{run_name} complete.\n",
        encoding="utf-8"
    )

    print("\nSaved:")
    print(" ", results_csv)
    print(" ", results_json)
    print(" ", complete_txt)

    if args.smoke:
        print("\nPhase 3 smoke test complete.")
    else:
        print("\nPhase 3 full white-box attack complete.")


if __name__ == "__main__":
    main()
