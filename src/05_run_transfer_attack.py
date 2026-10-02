from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARRAY_DIR = PROJECT_ROOT / "data" / "processed" / "baseline_arrays"
MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
SURROGATE_DIR = PROJECT_ROOT / "models" / "surrogate"
METRIC_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_DIR = PROJECT_ROOT / "config"
DOC_DIR = PROJECT_ROOT / "docs"

BASELINE_MANIFEST = METRIC_DIR / "baseline_freeze_manifest.csv"
PYTORCH_CHECKPOINT = MODEL_DIR / "pytorch_clean_baseline.pt"
XGBOOST_MODEL = MODEL_DIR / "xgboost_clean_baseline.json"
PREPROCESSOR_PATH = MODEL_DIR / "baseline_preprocessor.joblib"
FEASIBILITY_TABLE = METRIC_DIR / "feature_feasibility_table.csv"
SHARED_COHORT = ARRAY_DIR / "shared_correct_attack_test_indices.npy"
TRAIN_X = ARRAY_DIR / "train_X.npy"
TRAIN_Y = ARRAY_DIR / "train_y.npy"
VALIDATION_X = ARRAY_DIR / "validation_X.npy"
VALIDATION_Y = ARRAY_DIR / "validation_y.npy"
TEST_X = ARRAY_DIR / "test_X.npy"
TEST_RAW_X = ARRAY_DIR / "test_X_raw.npy"
TEST_Y = ARRAY_DIR / "test_y.npy"
TEST_LEDGER = METRIC_DIR / "pytorch_clean_test_predictions.csv.gz"
PHASE2_MARKER = METRIC_DIR / "phase2_feasibility_complete.txt"
PHASE3_MARKER = METRIC_DIR / "phase3_whitebox_complete.txt"
PHASE3_SUMMARY = METRIC_DIR / "phase3_whitebox_full" / "phase3_whitebox_attack_summary.csv"

SURROGATE_CHECKPOINT = SURROGATE_DIR / "phase4_transfer_surrogate.pt"
SURROGATE_METRICS = METRIC_DIR / "phase4_surrogate_training_metrics.json"


class TargetIDS(nn.Module):
    """Frozen Phase 1 target architecture."""

    def __init__(self, input_dim: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransferSurrogate(nn.Module):
    """Deliberately different surrogate architecture for transfer testing."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(0.20),
            nn.Linear(96, 48),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(48, 24),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(24, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ConstraintProjector:
    feature_names: list[str]
    direct_names: list[str]
    mask: torch.Tensor
    scaler_scale: torch.Tensor
    scaler_min: torch.Tensor
    feature_index: dict[str, int]

    def inverse_feature(self, scaled: torch.Tensor, index: int) -> torch.Tensor:
        scaled64 = scaled[:, index].to(dtype=torch.float64)
        return (scaled64 - self.scaler_min[index]) / self.scaler_scale[index]

    def transform_feature(
        self,
        raw: torch.Tensor,
        index: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        return (
            raw * self.scaler_scale[index] + self.scaler_min[index]
        ).to(dtype=output_dtype)

    @staticmethod
    def _integer_within_interval(
        candidate: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        lower_i = torch.ceil(lower)
        upper_i = torch.floor(upper)
        feasible = upper_i >= lower_i
        value = torch.round(candidate)
        value = torch.minimum(torch.maximum(value, lower_i), upper_i)
        return torch.where(feasible, value, fallback)

    def project(
        self,
        candidate_scaled: torch.Tensor,
        clean_scaled: torch.Tensor,
        clean_raw: torch.Tensor,
        epsilon: float,
        *,
        return_direct_raw: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        lower_scaled = torch.clamp(clean_scaled - float(epsilon), 0.0, 1.0)
        upper_scaled = torch.clamp(clean_scaled + float(epsilon), 0.0, 1.0)
        projected = clean_scaled.clone()
        output_dtype = projected.dtype
        clean_raw64 = clean_raw.to(dtype=torch.float64)
        raw_values: dict[str, torch.Tensor] = {}

        def increase_only(name: str, minimum: float) -> torch.Tensor:
            idx = self.feature_index[name]
            clean_value = clean_raw64[:, idx]
            candidate_raw = self.inverse_feature(candidate_scaled, idx)
            lower_raw = torch.maximum(
                torch.maximum(self.inverse_feature(lower_scaled, idx), clean_value),
                torch.full_like(clean_value, minimum),
            )
            upper_raw = self.inverse_feature(upper_scaled, idx)
            raw = self._integer_within_interval(
                candidate_raw, lower_raw, upper_raw, clean_value
            )
            projected[:, idx] = torch.clamp(
                self.transform_feature(raw, idx, output_dtype), 0.0, 1.0
            )
            raw_values[name] = raw
            return raw

        increase_only("Flow Duration", 0.0)
        total_fwd = increase_only("Total Fwd Packets", 1.0)
        increase_only("Total Length of Fwd Packets", 0.0)

        window_name = "Init_Win_bytes_forward"
        window_idx = self.feature_index[window_name]
        clean_window = clean_raw64[:, window_idx]
        candidate_window = self.inverse_feature(candidate_scaled, window_idx)
        lower_window = torch.maximum(
            self.inverse_feature(lower_scaled, window_idx),
            torch.zeros_like(clean_window),
        )
        upper_window = torch.minimum(
            self.inverse_feature(upper_scaled, window_idx),
            torch.full_like(clean_window, 65535.0),
        )
        window = self._integer_within_interval(
            candidate_window, lower_window, upper_window, clean_window
        )
        window = torch.where(clean_window < 0.0, clean_window, window)
        projected[:, window_idx] = torch.clamp(
            self.transform_feature(window, window_idx, output_dtype), 0.0, 1.0
        )
        raw_values[window_name] = window

        active_name = "act_data_pkt_fwd"
        active_idx = self.feature_index[active_name]
        clean_active = clean_raw64[:, active_idx]
        candidate_active = self.inverse_feature(candidate_scaled, active_idx)
        lower_active = torch.maximum(
            torch.maximum(self.inverse_feature(lower_scaled, active_idx), clean_active),
            torch.zeros_like(clean_active),
        )
        upper_active = torch.minimum(
            self.inverse_feature(upper_scaled, active_idx), total_fwd
        )
        active = self._integer_within_interval(
            candidate_active, lower_active, upper_active, clean_active
        )
        projected[:, active_idx] = torch.clamp(
            self.transform_feature(active, active_idx, output_dtype), 0.0, 1.0
        )
        raw_values[active_name] = active

        projected = torch.minimum(torch.maximum(projected, lower_scaled), upper_scaled)
        projected = torch.clamp(projected, 0.0, 1.0)
        projected = torch.where(self.mask > 0.5, projected, clean_scaled).detach()

        if not return_direct_raw:
            return projected

        direct_raw = torch.stack(
            [raw_values[name] for name in self.direct_names], dim=1
        ).detach()
        return projected, direct_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.05, 0.10],
    )
    parser.add_argument("--pgd-iterations", type=int, default=40)
    parser.add_argument("--pgd-step-ratio", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--surrogate-epochs", type=int, default=15)
    parser.add_argument("--surrogate-patience", type=int, default=4)
    parser.add_argument("--surrogate-train-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--retrain-surrogate", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def verify_phase1_baseline(output: Path) -> pd.DataFrame:
    manifest = pd.read_csv(BASELINE_MANIFEST)
    rows = []
    for row in manifest.itertuples(index=False):
        path = PROJECT_ROOT / str(row.file)
        exists = path.is_file()
        actual = sha256_file(path) if exists else ""
        expected = str(row.sha256).strip().lower()
        rows.append(
            {
                "file": str(row.file),
                "exists": exists,
                "hash_matches": exists and actual.lower() == expected,
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    failures = frame.loc[~frame["hash_matches"]]
    if not failures.empty:
        raise RuntimeError(
            "Frozen Phase 1 artifacts changed:\n"
            + failures[["file", "exists", "hash_matches"]].to_string(index=False)
        )
    return frame


def verify_phase3_evidence() -> None:
    if not PHASE3_MARKER.is_file() or not PHASE3_SUMMARY.is_file():
        raise FileNotFoundError("Validated Phase 3 completion evidence is missing.")
    phase3 = pd.read_csv(PHASE3_SUMMARY)
    constrained = phase3.loc[phase3["constraint_mode"] == "constrained"].copy()
    if len(phase3) != 20 or len(constrained) != 10:
        raise RuntimeError("Phase 3 result population is not the expected 20/10 design.")
    if "rows_with_any_direct_violation" not in constrained.columns:
        raise RuntimeError("Phase 3 corrected feasibility diagnostics are missing.")
    if int(pd.to_numeric(constrained["rows_with_any_direct_violation"]).sum()) != 0:
        raise RuntimeError("Phase 3 constrained evidence still contains violations.")


def load_target(device: torch.device) -> nn.Module:
    try:
        checkpoint = torch.load(
            PYTORCH_CHECKPOINT, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(PYTORCH_CHECKPOINT, map_location=device)
    spec = checkpoint.get("model_spec", {})
    input_dim = int(spec.get("input_dim", 77))
    dropout = float(spec.get("dropout", spec.get("dropout_rate", 0.30)))
    model = TargetIDS(input_dim, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_xgboost() -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(str(XGBOOST_MODEL))
    return model


def predict_torch(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = np.empty(len(x), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            batch = torch.as_tensor(x[start:end], dtype=torch.float32, device=device)
            output[start:end] = (
                torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
            )
    return output


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        generator=generator if shuffle else None,
    )


def surrogate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    train = optimizer is not None
    model.train(train)
    probs, targets = [], []
    total_loss = 0.0
    total_rows = 0
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()
            rows = len(yb)
            total_loss += float(loss.detach().cpu()) * rows
            total_rows += rows
            probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
            targets.append(yb.detach().cpu().numpy())
    return (
        total_loss / max(total_rows, 1),
        np.concatenate(probs).astype(np.float64),
        np.concatenate(targets).astype(np.int64),
    )


def train_or_load_surrogate(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    SURROGATE_DIR.mkdir(parents=True, exist_ok=True)

    if SURROGATE_CHECKPOINT.is_file() and not args.retrain_surrogate:
        try:
            checkpoint = torch.load(
                SURROGATE_CHECKPOINT, map_location=device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(SURROGATE_CHECKPOINT, map_location=device)
        model = TransferSurrogate(77).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        metrics = json.loads(SURROGATE_METRICS.read_text(encoding="utf-8"))
        return model, metrics

    x_train = np.load(TRAIN_X, mmap_mode="r")
    y_train = np.load(TRAIN_Y, mmap_mode="r")
    x_val = np.load(VALIDATION_X, mmap_mode="r")
    y_val = np.load(VALIDATION_Y, mmap_mode="r")

    all_indices = np.arange(len(y_train))
    subset_indices, _ = train_test_split(
        all_indices,
        train_size=float(args.surrogate_train_fraction),
        random_state=args.seed,
        stratify=np.asarray(y_train),
    )
    subset_indices = np.sort(subset_indices)
    x_subset = np.asarray(x_train[subset_indices], dtype=np.float32)
    y_subset = np.asarray(y_train[subset_indices], dtype=np.int64)
    x_validation = np.asarray(x_val, dtype=np.float32)
    y_validation = np.asarray(y_val, dtype=np.int64)

    counts = np.bincount(y_subset, minlength=2)
    weights = counts.sum() / (2.0 * counts.astype(np.float64))

    model = TransferSurrogate(77).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0012, weight_decay=0.0002
    )
    train_loader = make_loader(
        x_subset, y_subset, args.batch_size, True, args.seed
    )
    validation_loader = make_loader(
        x_validation, y_validation, args.batch_size, False, args.seed
    )

    best_f2 = -1.0
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, args.surrogate_epochs + 1):
        train_loss, train_prob, train_target = surrogate_epoch(
            model, train_loader, device, criterion, optimizer
        )
        val_loss, val_prob, val_target = surrogate_epoch(
            model, validation_loader, device, criterion, None
        )
        train_pred = (train_prob >= 0.5).astype(np.int64)
        val_pred = (val_prob >= 0.5).astype(np.int64)
        train_f2 = fbeta_score(
            train_target, train_pred, beta=2, pos_label=1, zero_division=0
        )
        val_f2 = fbeta_score(
            val_target, val_pred, beta=2, pos_label=1, zero_division=0
        )
        val_recall = recall_score(
            val_target, val_pred, pos_label=1, zero_division=0
        )
        val_precision = precision_score(
            val_target, val_pred, pos_label=1, zero_division=0
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "train_f2": train_f2,
                "validation_f2": val_f2,
                "validation_attack_recall": val_recall,
                "validation_attack_precision": val_precision,
            }
        )
        print(
            f"      surrogate epoch {epoch:02d}: "
            f"val_f2={val_f2:.6f} val_recall={val_recall:.6f}"
        )

        improved = (
            val_f2 > best_f2 + 1e-5
            or (abs(val_f2 - best_f2) <= 1e-5 and val_loss < best_loss - 1e-5)
        )
        if improved:
            best_f2 = float(val_f2)
            best_loss = float(val_loss)
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "input_dim": 77,
                    "seed": args.seed,
                    "train_fraction": args.surrogate_train_fraction,
                    "best_epoch": best_epoch,
                    "best_validation_f2": best_f2,
                },
                SURROGATE_CHECKPOINT,
            )
        else:
            stale += 1
        if stale >= args.surrogate_patience:
            break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(
        METRIC_DIR / "phase4_surrogate_training_history.csv", index=False
    )

    try:
        checkpoint = torch.load(
            SURROGATE_CHECKPOINT, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(SURROGATE_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    metrics = {
        "architecture": [77, 96, 48, 24, 2],
        "activation": "LeakyReLU",
        "dropout": 0.20,
        "seed": args.seed,
        "source_training_rows_available": int(len(y_train)),
        "surrogate_training_rows_used": int(len(y_subset)),
        "surrogate_training_fraction": float(args.surrogate_train_fraction),
        "validation_rows": int(len(y_validation)),
        "best_epoch": int(best_epoch),
        "best_validation_f2": float(best_f2),
        "training_seconds": round(time.perf_counter() - started, 3),
        "checkpoint_sha256": sha256_file(SURROGATE_CHECKPOINT),
        "target_weights_used_for_training": False,
        "target_gradients_used_for_training": False,
        "target_predictions_used_for_training": False,
    }
    write_json(SURROGATE_METRICS, metrics)
    return model, metrics


def build_projector(device: torch.device) -> ConstraintProjector:
    table = pd.read_csv(FEASIBILITY_TABLE)
    feature_names = table["feature"].astype(str).tolist()
    direct_names = table.loc[
        pd.to_numeric(table["conservative_mask"]) == 1, "feature"
    ].astype(str).tolist()
    expected = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Length of Fwd Packets",
        "Init_Win_bytes_forward",
        "act_data_pkt_fwd",
    ]
    if direct_names != expected:
        raise RuntimeError(
            f"Conservative Phase 2 mask changed. Expected {expected}, got {direct_names}."
        )
    mask_np = np.isin(feature_names, direct_names).astype(np.float32)[None, :]
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    scaler = preprocessor["scaler"]
    return ConstraintProjector(
        feature_names=feature_names,
        direct_names=direct_names,
        mask=torch.as_tensor(mask_np, dtype=torch.float32, device=device),
        scaler_scale=torch.as_tensor(
            np.asarray(scaler.scale_, dtype=np.float64),
            dtype=torch.float64,
            device=device,
        ),
        scaler_min=torch.as_tensor(
            np.asarray(scaler.min_, dtype=np.float64),
            dtype=torch.float64,
            device=device,
        ),
        feature_index={name: i for i, name in enumerate(feature_names)},
    )


def constrained_pgd(
    model: nn.Module,
    clean: np.ndarray,
    clean_raw: np.ndarray,
    epsilon: float,
    iterations: int,
    step_ratio: float,
    batch_size: int,
    device: torch.device,
    projector: ConstraintProjector,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.empty_like(clean, dtype=np.float32)
    direct_raw_output = np.empty(
        (len(clean), len(projector.direct_names)), dtype=np.float64
    )
    model.eval()
    step_size = max(float(epsilon) * float(step_ratio), 1e-6)

    for start in range(0, len(clean), batch_size):
        end = min(start + batch_size, len(clean))
        x0 = torch.as_tensor(
            clean[start:end], dtype=torch.float32, device=device
        )
        raw0 = torch.as_tensor(
            clean_raw[start:end], dtype=torch.float64, device=device
        )
        labels = torch.ones(len(x0), dtype=torch.long, device=device)
        adversarial = x0.detach().clone()
        lower = torch.clamp(x0 - float(epsilon), 0.0, 1.0)
        upper = torch.clamp(x0 + float(epsilon), 0.0, 1.0)

        for _ in range(iterations):
            adversarial = adversarial.detach().requires_grad_(True)
            loss = nn.functional.cross_entropy(model(adversarial), labels)
            gradient = torch.autograd.grad(loss, adversarial)[0]
            step = step_size * torch.sign(gradient) * projector.mask
            adversarial = adversarial.detach() + step
            adversarial = torch.minimum(torch.maximum(adversarial, lower), upper)
            adversarial = torch.clamp(adversarial, 0.0, 1.0)
            adversarial = projector.project(
                adversarial, x0, raw0, epsilon
            )

        adversarial, direct_raw = projector.project(
            adversarial, x0, raw0, epsilon, return_direct_raw=True
        )
        output[start:end] = adversarial.cpu().numpy().astype(np.float32)
        direct_raw_output[start:end] = direct_raw.cpu().numpy().astype(np.float64)

    return output, direct_raw_output


def constraint_diagnostics(
    clean: np.ndarray,
    clean_raw: np.ndarray,
    adversarial: np.ndarray,
    direct_raw: np.ndarray,
    epsilon: float,
    projector: ConstraintProjector,
) -> dict[str, Any]:
    mask = projector.mask.detach().cpu().numpy()[0].astype(bool)
    frozen_delta = np.abs(adversarial[:, ~mask] - clean[:, ~mask])
    all_delta = np.abs(adversarial - clean)

    frozen_change = np.any(frozen_delta > 1e-7, axis=1)
    eps_fail = np.max(all_delta, axis=1) > float(epsilon) + 2e-6
    bounds_fail = np.any((adversarial < -1e-7) | (adversarial > 1.0000001), axis=1)
    fractional_fail = np.any(
        np.abs(direct_raw - np.round(direct_raw)) > 1e-8, axis=1
    )

    raw_clean_direct = np.column_stack(
        [
            clean_raw[:, projector.feature_index[name]]
            for name in projector.direct_names
        ]
    ).astype(np.float64)

    # Columns 0,1,2,4 are increase-only. Window is limited but may move either way.
    increase_columns = [0, 1, 2, 4]
    direction_fail = np.any(
        direct_raw[:, increase_columns]
        < raw_clean_direct[:, increase_columns] - 1e-8,
        axis=1,
    )

    duration = direct_raw[:, 0]
    total_fwd = direct_raw[:, 1]
    total_len = direct_raw[:, 2]
    window = direct_raw[:, 3]
    active = direct_raw[:, 4]
    window_valid = (window == -1) | ((window >= 0) & (window <= 65535))
    domain_fail = (
        (duration < 0)
        | (total_fwd < 1)
        | (total_len < 0)
        | (~window_valid)
        | (active < 0)
        | (active > total_fwd)
    )

    any_fail = (
        frozen_change
        | eps_fail
        | bounds_fail
        | fractional_fail
        | direction_fail
        | domain_fail
    )

    return {
        "rows_with_frozen_feature_changes": int(frozen_change.sum()),
        "rows_exceeding_epsilon": int(eps_fail.sum()),
        "rows_outside_scaled_bounds": int(bounds_fail.sum()),
        "rows_with_fractional_direct_values": int(fractional_fail.sum()),
        "rows_with_direction_violations": int(direction_fail.sum()),
        "rows_with_domain_violations": int(domain_fail.sum()),
        "rows_with_any_direct_violation": int(any_fail.sum()),
        "direct_violation_rate": float(any_fail.mean()),
        "direct_validation_basis": "projected_raw_contract_v2",
        "mean_linf_scaled": float(np.mean(np.max(all_delta, axis=1))),
        "max_linf_scaled": float(np.max(np.max(all_delta, axis=1))),
        "mean_changed_features": float(np.mean(np.sum(all_delta > 1e-7, axis=1))),
    }


def select_cohort(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    shared_indices = np.load(SHARED_COHORT, allow_pickle=False).astype(np.int64)
    test_x = np.load(TEST_X, mmap_mode="r")
    test_raw = np.load(TEST_RAW_X, mmap_mode="r")
    test_y = np.load(TEST_Y, mmap_mode="r")
    ledger = pd.read_csv(TEST_LEDGER, compression="gzip").sort_values(
        "SplitPosition"
    ).reset_index(drop=True)

    if np.any(np.asarray(test_y[shared_indices]) != 1):
        raise RuntimeError("Shared Phase 2 cohort contains non-attack rows.")

    selected = shared_indices
    if args.max_rows and args.max_rows < len(selected):
        rng = np.random.default_rng(args.seed + 91)
        chosen = np.sort(
            rng.choice(len(selected), size=args.max_rows, replace=False)
        )
        selected = selected[chosen]

    x = np.asarray(test_x[selected], dtype=np.float32)
    x_raw = np.asarray(test_raw[selected], dtype=np.float64)
    metadata = ledger.iloc[selected].copy().reset_index(drop=True)
    metadata["TestPosition"] = selected
    return selected, x, x_raw, metadata


def phase3_reference() -> dict[float, float]:
    frame = pd.read_csv(PHASE3_SUMMARY)
    subset = frame.loc[
        (frame["attack_method"] == "PGD")
        & (frame["constraint_mode"] == "constrained")
    ]
    return {
        float(row.epsilon): float(row.evasion_rate)
        for row in subset.itertuples(index=False)
    }


def save_figures(summary: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(
        summary["epsilon"],
        summary["surrogate_whitebox_evasion_rate"],
        marker="o",
        label="Surrogate white-box",
    )
    ax.plot(
        summary["epsilon"],
        summary["pytorch_transfer_evasion_rate"],
        marker="o",
        label="Transfer → PyTorch",
    )
    ax.plot(
        summary["epsilon"],
        summary["xgboost_transfer_evasion_rate"],
        marker="o",
        label="Transfer → XGBoost",
    )
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Evasion rate")
    ax.set_title("Phase 4A Transfer Attack Evasion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "transfer_evasion_vs_epsilon.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(
        summary["epsilon"],
        summary["pytorch_transfer_success_ratio"],
        marker="o",
        label="PyTorch / surrogate",
    )
    ax.plot(
        summary["epsilon"],
        summary["xgboost_transfer_success_ratio"],
        marker="o",
        label="XGBoost / surrogate",
    )
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Transfer success ratio")
    ax.set_title("Transfer Success Relative to Surrogate White-Box Evasion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "transfer_success_ratio.png", dpi=180)
    plt.close(fig)


def artifact_manifest(metric_dir: Path, figure_dir: Path, methodology: Path) -> pd.DataFrame:
    manifest_path = metric_dir / "phase4a_artifact_manifest.csv"
    files = [
        path
        for path in metric_dir.iterdir()
        if path.is_file() and path.name != manifest_path.name
    ]
    files.extend(path for path in figure_dir.iterdir() if path.is_file())
    files.append(methodology)
    rows = []
    for path in sorted(set(files)):
        rows.append(
            {
                "file": str(path.relative_to(PROJECT_ROOT)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(manifest_path, index=False)
    return frame


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    required = [
        BASELINE_MANIFEST,
        PYTORCH_CHECKPOINT,
        XGBOOST_MODEL,
        PREPROCESSOR_PATH,
        FEASIBILITY_TABLE,
        SHARED_COHORT,
        TRAIN_X,
        TRAIN_Y,
        VALIDATION_X,
        VALIDATION_Y,
        TEST_X,
        TEST_RAW_X,
        TEST_Y,
        TEST_LEDGER,
        PHASE2_MARKER,
        PHASE3_MARKER,
        PHASE3_SUMMARY,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required prior-phase artifacts are missing:\n" + "\n".join(missing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_dir = METRIC_DIR / f"phase4a_transfer_{args.run_name}"
    figure_dir = FIGURE_DIR / f"phase4a_transfer_{args.run_name}"
    metric_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("ADVERSARIAL IDS PHASE 4A — TRANSFER ATTACK")
    print("=" * 78)
    print(f"Run name:          {args.run_name}")
    print(f"Device:            {device}")
    print(f"Epsilon values:    {args.epsilons}")
    print(f"PGD iterations:    {args.pgd_iterations}")
    print("Attacker knowledge: surrogate only; no target gradients")

    print("[1/8] Verifying frozen Phase 1 and validated Phase 3 evidence...")
    before = verify_phase1_baseline(metric_dir / "baseline_integrity_before.csv")
    verify_phase3_evidence()
    print(f"      Phase 1 artifacts verified: {len(before)}")

    print("[2/8] Training or loading the independent surrogate...")
    surrogate, surrogate_training = train_or_load_surrogate(args, device)
    print(
        f"      Surrogate training rows used: "
        f"{surrogate_training['surrogate_training_rows_used']:,}"
    )
    print(f"      Best validation F2: {surrogate_training['best_validation_f2']:.6f}")

    print("[3/8] Loading frozen target models...")
    target_torch = load_target(device)
    target_xgb = load_xgboost()

    print("[4/8] Loading the shared target-correct attack cohort...")
    selected_positions, clean_x, clean_raw, metadata = select_cohort(args)
    print(f"      Rows evaluated: {len(clean_x):,}")
    print(f"      Full shared cohort: {len(np.load(SHARED_COHORT)):,}")

    projector = build_projector(device)
    print(
        "      Conservative fields: "
        + ", ".join(projector.direct_names)
    )

    surrogate_clean_prob = predict_torch(
        surrogate, clean_x, device, args.batch_size
    )
    target_clean_prob = predict_torch(
        target_torch, clean_x, device, args.batch_size
    )
    xgb_clean_prob = target_xgb.predict_proba(clean_x)[:, 1].astype(np.float64)

    if np.any(target_clean_prob < 0.5) or np.any(xgb_clean_prob < 0.5):
        raise RuntimeError(
            "Shared cohort no longer represents rows correctly detected by both targets."
        )

    surrogate_clean_attack = surrogate_clean_prob >= 0.5
    print(
        f"      Surrogate clean attack coverage: "
        f"{surrogate_clean_attack.sum():,}/{len(clean_x):,} "
        f"({surrogate_clean_attack.mean():.2%})"
    )

    phase3_pgd = phase3_reference()
    summary_rows: list[dict[str, Any]] = []
    ledger_rows: list[pd.DataFrame] = []
    per_label_rows: list[dict[str, Any]] = []

    print("[5/8] Crafting constrained PGD on the surrogate and testing transfer...")
    for run_number, epsilon in enumerate(args.epsilons, start=1):
        started = time.perf_counter()
        print(
            f"      [{run_number}/{len(args.epsilons)}] "
            f"constrained surrogate PGD eps={epsilon:.4f}"
        )
        adversarial, direct_raw = constrained_pgd(
            surrogate,
            clean_x,
            clean_raw,
            float(epsilon),
            args.pgd_iterations,
            args.pgd_step_ratio,
            args.batch_size,
            device,
            projector,
        )
        diagnostics = constraint_diagnostics(
            clean_x,
            clean_raw,
            adversarial,
            direct_raw,
            float(epsilon),
            projector,
        )
        if diagnostics["rows_with_any_direct_violation"] != 0:
            raise RuntimeError(
                f"Constraint validation failed at epsilon {epsilon}: {diagnostics}"
            )

        surrogate_adv_prob = predict_torch(
            surrogate, adversarial, device, args.batch_size
        )
        pytorch_adv_prob = predict_torch(
            target_torch, adversarial, device, args.batch_size
        )
        xgb_adv_prob = target_xgb.predict_proba(adversarial)[:, 1].astype(np.float64)

        surrogate_evasion_mask = surrogate_clean_attack & (surrogate_adv_prob < 0.5)
        pytorch_transfer_mask = pytorch_adv_prob < 0.5
        xgb_transfer_mask = xgb_adv_prob < 0.5

        surrogate_denominator = int(surrogate_clean_attack.sum())
        surrogate_rate = (
            float(surrogate_evasion_mask.sum() / surrogate_denominator)
            if surrogate_denominator
            else 0.0
        )
        pytorch_rate = float(pytorch_transfer_mask.mean())
        xgb_rate = float(xgb_transfer_mask.mean())

        phase3_rate = phase3_pgd.get(float(epsilon), float("nan"))

        summary_row = {
            "epsilon": float(epsilon),
            "cohort_rows": int(len(clean_x)),
            "surrogate_clean_attack_rows": surrogate_denominator,
            "surrogate_clean_attack_coverage": float(surrogate_clean_attack.mean()),
            "surrogate_whitebox_evasion_rows": int(surrogate_evasion_mask.sum()),
            "surrogate_whitebox_evasion_rate": surrogate_rate,
            "pytorch_transfer_evasion_rows": int(pytorch_transfer_mask.sum()),
            "pytorch_transfer_evasion_rate": pytorch_rate,
            "xgboost_transfer_evasion_rows": int(xgb_transfer_mask.sum()),
            "xgboost_transfer_evasion_rate": xgb_rate,
            "pytorch_post_transfer_attack_recall": float(1.0 - pytorch_rate),
            "xgboost_post_transfer_attack_recall": float(1.0 - xgb_rate),
            "pytorch_transfer_success_ratio": (
                float(pytorch_rate / surrogate_rate) if surrogate_rate > 0 else 0.0
            ),
            "xgboost_transfer_success_ratio": (
                float(xgb_rate / surrogate_rate) if surrogate_rate > 0 else 0.0
            ),
            "phase3_pytorch_whitebox_pgd_evasion_rate": phase3_rate,
            "pytorch_whitebox_minus_transfer_gap": (
                float(phase3_rate - pytorch_rate)
                if np.isfinite(phase3_rate)
                else float("nan")
            ),
            "mean_clean_surrogate_attack_probability": float(
                surrogate_clean_prob.mean()
            ),
            "mean_adversarial_surrogate_attack_probability": float(
                surrogate_adv_prob.mean()
            ),
            "mean_adversarial_pytorch_attack_probability": float(
                pytorch_adv_prob.mean()
            ),
            "mean_adversarial_xgboost_attack_probability": float(
                xgb_adv_prob.mean()
            ),
            **diagnostics,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        summary_rows.append(summary_row)

        ledger = metadata[
            ["TestPosition", "RowID", "CanonicalLabel", "BinaryLabel"]
        ].copy()
        ledger["epsilon"] = float(epsilon)
        ledger["SurrogateCleanAttackProbability"] = surrogate_clean_prob
        ledger["SurrogateAdversarialAttackProbability"] = surrogate_adv_prob
        ledger["PyTorchCleanAttackProbability"] = target_clean_prob
        ledger["PyTorchAdversarialAttackProbability"] = pytorch_adv_prob
        ledger["XGBoostCleanAttackProbability"] = xgb_clean_prob
        ledger["XGBoostAdversarialAttackProbability"] = xgb_adv_prob
        ledger["SurrogateOriginallyCorrect"] = surrogate_clean_attack
        ledger["SurrogateWhiteBoxEvaded"] = surrogate_evasion_mask
        ledger["PyTorchTransferEvaded"] = pytorch_transfer_mask
        ledger["XGBoostTransferEvaded"] = xgb_transfer_mask
        for column, name in enumerate(projector.direct_names):
            ledger[f"ProjectedRaw_{name}"] = direct_raw[:, column]
        ledger_rows.append(ledger)

        labels = metadata["CanonicalLabel"].astype(str).to_numpy()
        for label in sorted(np.unique(labels)):
            mask = labels == label
            label_rows = int(mask.sum())
            per_label_rows.append(
                {
                    "epsilon": float(epsilon),
                    "CanonicalLabel": label,
                    "cohort_rows": label_rows,
                    "surrogate_clean_attack_rows": int(
                        (surrogate_clean_attack & mask).sum()
                    ),
                    "surrogate_whitebox_evasion_rows": int(
                        (surrogate_evasion_mask & mask).sum()
                    ),
                    "pytorch_transfer_evasion_rows": int(
                        (pytorch_transfer_mask & mask).sum()
                    ),
                    "pytorch_transfer_evasion_rate": float(
                        pytorch_transfer_mask[mask].mean()
                    ),
                    "xgboost_transfer_evasion_rows": int(
                        (xgb_transfer_mask & mask).sum()
                    ),
                    "xgboost_transfer_evasion_rate": float(
                        xgb_transfer_mask[mask].mean()
                    ),
                }
            )

    print("[6/8] Writing metrics, ledgers, figures, and methodology...")
    summary = pd.DataFrame(summary_rows).sort_values("epsilon")
    summary.to_csv(metric_dir / "phase4a_transfer_summary.csv", index=False)
    pd.DataFrame(per_label_rows).to_csv(
        metric_dir / "phase4a_transfer_per_label.csv", index=False
    )
    pd.concat(ledger_rows, ignore_index=True).to_csv(
        metric_dir / "phase4a_transfer_prediction_ledger.csv.gz",
        index=False,
        compression="gzip",
    )
    save_figures(summary, figure_dir)

    methodology = DOC_DIR / f"phase4a_transfer_methodology_{args.run_name}.md"
    methodology.write_text(
        "\n".join(
            [
                "# Phase 4A Transfer Attack Methodology",
                "",
                "The attacker trains an independent surrogate neural network and never uses the protected target models' gradients or weights.",
                "",
                f"- Run: `{args.run_name}`",
                f"- Surrogate architecture: 77 → 96 → 48 → 24 → 2",
                f"- Surrogate training fraction of frozen Phase 1 training set: {args.surrogate_train_fraction:.2f}",
                f"- Surrogate seed: {args.seed}",
                f"- Shared target-correct cohort rows in this run: {len(clean_x):,}",
                f"- PGD iterations: {args.pgd_iterations}",
                f"- Epsilon grid: {', '.join(str(v) for v in args.epsilons)}",
                f"- Feasible direct features: {', '.join(projector.direct_names)}",
                "- The same adversarial rows crafted against the surrogate are passed unchanged to the frozen PyTorch and XGBoost targets.",
                "- Transfer evasion is attack→benign among rows that both target models classified correctly before perturbation.",
                "- Feasibility is validated with the projected raw contract used by corrected Phase 3.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary_text = metric_dir / "phase4a_transfer_summary.txt"
    lines = [
        "ADVERSARIAL IDS PHASE 4A — TRANSFER ATTACK SUMMARY",
        "=" * 78,
        f"Run name:                         {args.run_name}",
        f"Rows evaluated:                   {len(clean_x):,}",
        f"Full shared target-correct cohort:{len(np.load(SHARED_COHORT)):,}",
        f"Surrogate clean attack coverage:  {surrogate_clean_attack.mean():.6f}",
        f"Constraint failures:              {int(summary['rows_with_any_direct_violation'].sum())}",
        "",
        "Per-epsilon transfer results:",
    ]
    for row in summary.itertuples(index=False):
        lines.extend(
            [
                (
                    f"  eps={row.epsilon:.3f} | "
                    f"surrogate={row.surrogate_whitebox_evasion_rate:.6f} | "
                    f"PyTorch transfer={row.pytorch_transfer_evasion_rate:.6f} | "
                    f"XGBoost transfer={row.xgboost_transfer_evasion_rate:.6f}"
                )
            ]
        )
    summary_text.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[7/8] Re-verifying frozen Phase 1 artifacts...")
    after = verify_phase1_baseline(metric_dir / "baseline_integrity_after.csv")
    print(f"      Verified artifacts after transfer test: {len(after)}")

    print("[8/8] Freezing Phase 4A evidence...")
    manifest = artifact_manifest(metric_dir, figure_dir, methodology)
    (METRIC_DIR / f"phase4a_transfer_{args.run_name}_complete.txt").write_text(
        "PASS: Phase 4A transfer attack completed with zero feasibility violations.\n",
        encoding="utf-8",
    )

    print()
    print("PASS: Phase 4A transfer attack completed.")
    print(f"Summary:   {summary_text}")
    print(f"Metrics:   {metric_dir / 'phase4a_transfer_summary.csv'}")
    print(f"Figures:   {figure_dir}")
    print(f"Artifacts frozen: {len(manifest)}")


if __name__ == "__main__":
    main()
