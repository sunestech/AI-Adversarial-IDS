from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import HopSkipJump


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


def load_pytorch_model(device):
    checkpoint_path = MODEL_DIR / "pytorch_clean_baseline.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing PyTorch baseline: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_dim = int(checkpoint["input_dim"])

    model = IDSClassifier(input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, input_dim


def predict_torch(model, X, device, batch_size=1024):
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


def project_to_constraints(X_orig, X_adv_raw, eps, mask):
    # Apply epsilon projection.
    X_adv = X_orig + np.clip(X_adv_raw - X_orig, -eps, eps)

    # Freeze non-attacker-controllable features.
    X_adv[:, mask == 0] = X_orig[:, mask == 0]

    # Keep scaled feature space valid.
    X_adv = np.clip(X_adv, 0.0, 1.0)

    return X_adv.astype(np.float32)


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
        ARRAYS_DIR / "X_test_scaled.npy",
        ARRAYS_DIR / "y_test.npy",
        ARRAYS_DIR / "shared_correct_attack_test_indices.npy",
        MODEL_DIR / "pytorch_clean_baseline.pt",
        METRICS_DIR / "feature_feasibility_table.csv",
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    X_test = np.load(ARRAYS_DIR / "X_test_scaled.npy").astype(np.float32)
    y_test = np.load(ARRAYS_DIR / "y_test.npy").astype(int)
    shared_indices = np.load(ARRAYS_DIR / "shared_correct_attack_test_indices.npy").astype(int)

    feasibility = pd.read_csv(METRICS_DIR / "feature_feasibility_table.csv")
    mask = feasibility["perturbation_allowed"].astype(int).to_numpy()

    model, input_dim = load_pytorch_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(input_dim,),
        nb_classes=2,
        clip_values=(0.0, 1.0)
    )

    if args.smoke:
        run_name = "phase4b_hopskipjump_smoke"
        selected_indices = shared_indices[:32]
        max_iter = 5
        max_eval = 100
        init_eval = 10
        eps_values = [0.05, 0.10]
    else:
        run_name = "phase4b_hopskipjump_full"
        selected_indices = shared_indices[:300]
        max_iter = 20
        max_eval = 500
        init_eval = 50
        eps_values = [0.05, 0.10]

    X_attack = X_test[selected_indices]
    y_attack = y_test[selected_indices]

    print("Run name:", run_name)
    print("Attack rows:", len(X_attack))
    print("HopSkipJump max_iter:", max_iter)
    print("HopSkipJump max_eval:", max_eval)
    print("HopSkipJump init_eval:", init_eval)

    base_pred, base_scores = predict_torch(model, X_attack, device)
    base_detected = int((base_pred == 1).sum())

    print("Baseline detected attacks:", base_detected)

    attack = HopSkipJump(
        classifier=classifier,
        targeted=False,
        max_iter=max_iter,
        max_eval=max_eval,
        init_eval=init_eval,
        verbose=True
    )

    print("\nGenerating raw HopSkipJump adversarial examples...")
    X_adv_raw = attack.generate(x=X_attack)

    raw_pred, _ = predict_torch(model, X_adv_raw.astype(np.float32), device)
    raw_evasion_rate = float((raw_pred == 0).mean())

    print("Raw HSJ evasion rate:", raw_evasion_rate)

    rows = []

    for eps in eps_values:
        X_adv = project_to_constraints(
            X_orig=X_attack,
            X_adv_raw=X_adv_raw,
            eps=eps,
            mask=mask
        )

        adv_pred, adv_scores = predict_torch(model, X_adv, device)

        evasion_count = int((adv_pred == 0).sum())
        evasion_rate = float(evasion_count / len(X_attack))
        post_attack_recall = float((adv_pred == 1).mean())

        validation = validate_constraints(X_attack, X_adv, eps, mask)

        row = {
            "run_name": run_name,
            "attack_method": "HopSkipJump",
            "target_model": "PyTorch",
            "attack_rows": int(len(X_attack)),
            "max_iter": int(max_iter),
            "max_eval": int(max_eval),
            "init_eval": int(init_eval),
            "configured_query_budget_per_sample_approx": int(max_iter * max_eval),
            "epsilon_projection": float(eps),
            "baseline_detected_attack_rows": int(base_detected),
            "raw_unprojected_evasion_rate": raw_evasion_rate,
            "constrained_projected_evasion_count": evasion_count,
            "constrained_projected_evasion_rate": evasion_rate,
            "post_attack_recall": post_attack_recall,
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
            f"projected_evasion={evasion_rate:.4f} | "
            f"post_attack_recall={post_attack_recall:.4f} | "
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

    print("\nPhase 4B HopSkipJump attack complete.")


if __name__ == "__main__":
    main()
