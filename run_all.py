"""Run 60 DDAE experiments while preserving the original run.py pipeline.

Place this file next to run.py and execute: python run_all.py
The original src modules and config remain unchanged. Results are appended
immediately after each experiment; failed experiments do not stop the sweep.
"""

import argparse
import copy
import csv
import hashlib
import json
import re
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data import load_data, split_data
from src.model import DDAE
from src.utils import (
    evaluate_anomaly_detection, get_batch_size, normalize_data, set_seed,
)

# Only these four experiment axes differ between runs.
DATASETS = ("campaign", "annthyroid", "satimage-2")
SETTINGS = ("unsupervised", "semi-supervised")
SEEDS = (111, 222, 333, 444, 555)
TIMESTEPS = (50, 100)
FIELDS = (
    "dataset", "filename", "setting", "seed", "T", "config_id",
    "AP", "PR-AUC", "ROC-AUC", "n_train", "n_test", "batch_size",
    "total_seconds", "final_predict_seconds", "status", "error",
)


def parse_args():
    """Resolve default paths relative to this script, not the shell directory."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "src/config.yaml")
    parser.add_argument("--data-dir", type=Path, default=root / "ad_bench")
    parser.add_argument("--output-dir", type=Path, default=root / "output/ddae_runs")
    parser.add_argument("--resume", action="store_true", help="Skip successful runs with identical resolved config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate files and print the 60-run manifest without training.")
    return parser.parse_args()


def canonical(name):
    """Match ADBench names, including numeric prefixes and punctuation."""
    name = re.sub(r"^\d+[_-]", "", name.lower())
    return re.sub(r"[^a-z0-9]", "", name)


def resolve_datasets(data_dir):
    """Fail early rather than silently choosing an ambiguous dataset file."""
    mapping = {}
    for dataset in DATASETS:
        matches = [p for p in data_dir.glob("*.npz") if canonical(p.stem) == canonical(dataset)]
        if len(matches) != 1:
            raise ValueError(f"{dataset}: expected one .npz file, found {[p.name for p in matches]}")
        mapping[dataset] = matches[0].name
    return mapping


def validate_data(X, y):
    """Basic checks only: never reshape, filter, or otherwise change the data."""
    if X.ndim != 2 or not X.size or y.ndim != 1 or len(X) != len(y):
        raise ValueError("Expected nonempty 2D X and matching 1D y")
    if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X) or not np.isfinite(X).all():
        raise ValueError("X must contain finite real numeric features")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("y must contain both normal (0) and anomaly (1) labels")


def run_experiment(config):
    """Mirror original run.py, including RNG order and final prediction call."""
    start = time.perf_counter()
    seed = config.get("seed", 111)

    # Reset once at the beginning of EACH experiment, exactly as run.py does.
    set_seed(seed)
    X, y = load_data(config["data"]["path"], config["data"]["name"])
    validate_data(X, y)

    # Preserve original preprocessing: fit StandardScaler on ALL X before split.
    X = normalize_data(X)
    x_train, x_test, y_train, y_test = split_data(
        X, y, train_setting=config["train"]["setting"], random_state=seed,
    )
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Original policy uses full dataset size, NOT train size or YAML batch_size.
    batch_size = get_batch_size(X.shape[0])
    model = DDAE(
        input_dim=x_train.shape[1],
        hidden_dim=config["model"]["hidden_dim"],
        activation=config["model"]["activation"],
        num_timesteps=config["diffusion"]["num_timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        scheduler=config["diffusion"]["scheduler"],
        time_emb_dim=config["diffusion"]["time_emb_dim"],
        time_emb_type=config["diffusion"]["time_emb_type"],
        epochs=config["train"]["epochs"],
        batch_size=batch_size,
        learning_rate=config["train"]["lr"],
        eval_epochs=config["train"]["eval_epochs"],
    )
    # Do not add device selection, bootstrap, custom initialization, or extra
    # predictions: all would change original execution behavior or RNG usage.
    print(f"Batch size: {batch_size}", flush=True)
    model.fit(x_train, x_test, y_train, y_test)

    # Keep the extra final predict from original run.py, not the epoch-100 log.
    predict_start = time.perf_counter()
    scores = model.predict(x_test)
    predict_seconds = time.perf_counter() - predict_start
    results = evaluate_anomaly_detection(scores=scores.numpy(), labels=y_test.numpy())
    metadata = {
        "n_train": len(x_train), "n_test": len(x_test), "batch_size": batch_size,
        "total_seconds": time.perf_counter() - start,
        "final_predict_seconds": predict_seconds,
    }
    return {key: float(value) for key, value in results.items()}, metadata


def config_id(config):
    """Bind resume to all resolved config fields, not just the four axes."""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def successful_ids(path):
    """Failed rows are retained for audit but never skipped on resume."""
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Existing CSV has a different schema; use a new output directory")
        return {row["config_id"] for row in reader if row["status"] == "SUCCESS"}


def append_result(path, row):
    """Close the CSV after each row so completed results survive interruption."""
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    with args.config.open(encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    if not isinstance(base, dict):
        raise ValueError("Config must be a YAML mapping")
    mapping = resolve_datasets(args.data_dir)
    experiments = []

    # Deep copies prevent one run's overrides from leaking into another run.
    for dataset in DATASETS:
        for setting in SETTINGS:
            for seed in SEEDS:
                for T in TIMESTEPS:
                    config = copy.deepcopy(base)
                    config["data"]["path"] = str(args.data_dir)
                    config["data"]["name"] = mapping[dataset]
                    config["train"]["setting"] = setting
                    config["seed"] = seed
                    config["diffusion"]["num_timesteps"] = T
                    experiments.append((dataset, config, config_id(config)))
    if len(experiments) != 60 or len({item[2] for item in experiments}) != 60:
        raise ValueError("Expected exactly 60 distinct experiments")

    # Dry run does not instantiate models, train, or create output files.
    if args.dry_run:
        for dataset, config, identity in experiments:
            print(dataset, config["data"]["name"], config["train"]["setting"], config["seed"], config["diffusion"]["num_timesteps"])
        print(f"Total: {len(experiments)} experiments; data contents are checked during actual runs.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "results.csv"
    if csv_path.exists() and not args.resume:
        raise FileExistsError("results.csv already exists: use --resume or a new --output-dir")
    completed = successful_ids(csv_path) if args.resume else set()
    configs_dir = args.output_dir / "configs"
    errors_dir = args.output_dir / "errors"
    configs_dir.mkdir(exist_ok=True)
    errors_dir.mkdir(exist_ok=True)
    failures = 0

    for number, (dataset, config, identity) in enumerate(experiments, 1):
        label = f"{dataset}_{config['train']['setting']}_seed{config['seed']}_T{config['diffusion']['num_timesteps']}"
        if identity in completed:
            print(f"[{number}/60] SKIP {label}", flush=True)
            continue
        print(f"[{number}/60] RUN {label}", flush=True)
        # Archive exact settings; no source config file is modified.
        (configs_dir / f"{label}_{identity[:12]}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        row = {
            "dataset": dataset, "filename": config["data"]["name"],
            "setting": config["train"]["setting"], "seed": config["seed"],
            "T": config["diffusion"]["num_timesteps"], "config_id": identity,
        }
        start = time.perf_counter()
        try:
            results, metadata = run_experiment(config)
            row.update(results)
            row.update(metadata)
            row.update(status="SUCCESS", error="")
            completed.add(identity)
            print(f"SUCCESS {label}: {results}", flush=True)
        except Exception as exc:
            # KeyboardInterrupt is intentionally not caught: Ctrl+C stops cleanly.
            failures += 1
            row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", total_seconds=time.perf_counter() - start)
            (errors_dir / f"{label}_{identity[:12]}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"FAILED {label}: {exc}", flush=True)
        append_result(csv_path, row)
    print(f"Finished: {len(completed)}/60 successful configurations; {failures} failures this invocation.")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
