"""
run.py — Pipeline Runner
========================
Convenience entry-point for the Unified HSI Medical Framework.

Usage:
    python run.py                       # start backend server
    python run.py --train tumor         # train tumor models
    python run.py --train perfusion     # train perfusion models
    python run.py --train both          # train both
    python run.py --predict tumor  path/to/file.hdr [path/to/file.raw]
    python run.py --predict perfusion  path/to/file.npy
    python run.py --health              # check model readiness
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import (
    LOG_FORMAT, LOG_LEVEL, LOGS_DIR,
    TRAINED_MODELS_DIR, EVAL_METRICS_DIR, PREDICTION_RESULTS_DIR,
)

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOGS_DIR / "run.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─── Actions ─────────────────────────────────────────────────────────────────

def action_serve():
    """Start the FastAPI backend server."""
    import uvicorn
    logger.info("Starting HSI Medical Framework backend on http://0.0.0.0:8000")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


def action_train(task: str):
    """Train models for 'tumor', 'perfusion', or 'both'."""
    if task in ("tumor", "both"):
        logger.info("=== Training tumor models ===")
        from ml_pipeline.training.train_tumor import train_tumor_models
        train_tumor_models()

    if task in ("perfusion", "both"):
        logger.info("=== Training perfusion models ===")
        from ml_pipeline.training.train_perfusion import train_perfusion_models
        train_perfusion_models()


def action_predict(task: str, file_path: str, raw_path: str = None):
    """Run prediction on a file from the command line."""
    from ml_pipeline.prediction.predict import predict_tumor, predict_perfusion
    import json

    logger.info("Running %s prediction on: %s", task, file_path)

    if task == "tumor":
        result = predict_tumor(file_path, raw_path)
    else:
        result = predict_perfusion(file_path, raw_path)

    # Pretty-print (omit heatmap data for readability)
    display = {k: v for k, v in result.items() if k != "heatmap"}
    print(json.dumps(display, indent=2))


def action_health():
    """Check model file readiness."""
    # NOTE: Use only ASCII characters here so this function works correctly
    # on all platforms — including Windows PowerShell / cp1252 terminals and
    # Docker/CI containers where stdout encoding may not be UTF-8.
    tasks  = ["tumor", "perfusion"]
    models = ["rf", "svm", "scaler", "pca"]

    print("\n-- Model Health Check ------------------------------------------")
    all_ok = True
    for task in tasks:
        for m in models:
            p  = TRAINED_MODELS_DIR / f"{task}_{m}.pkl"
            ok = p.exists()
            status = "OK" if ok else "MISSING"
            if not ok:
                all_ok = False
            print(f"  [{status}]  {p.name}")

    print("\n-- Evaluation Metrics ------------------------------------------")
    for task in tasks:
        p  = EVAL_METRICS_DIR / f"{task}_metrics.json"
        ok = p.exists()
        print(f"  [{'OK' if ok else 'MISSING'}]  {p.name}")

    print("\n-- Prediction Results ------------------------------------------")
    results = list(PREDICTION_RESULTS_DIR.glob("*.json"))
    print(f"  {len(results)} saved prediction(s)")

    print()
    if all_ok:
        print("[OK] All models ready. Run: python run.py  to start server.")
    else:
        print("[!!] Some models missing. Run:")
        print("    python run.py --train both")
    print()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified HSI Medical Framework — Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--train", choices=["tumor", "perfusion", "both"],
        metavar="TASK",
        help="Train models: tumor | perfusion | both",
    )
    group.add_argument(
        "--predict", nargs="+",
        metavar="ARG",
        help="Predict: tumor <file.hdr> [<file.raw>]  OR  perfusion <file.npy>",
    )
    group.add_argument(
        "--health", action="store_true",
        help="Check model file readiness",
    )

    args = parser.parse_args()

    if args.train:
        action_train(args.train)

    elif args.predict:
        if len(args.predict) < 2:
            parser.error("--predict requires: TASK FILE_PATH [RAW_PATH]")
        task      = args.predict[0]
        file_path = args.predict[1]
        raw_path  = args.predict[2] if len(args.predict) > 2 else None
        if task not in ("tumor", "perfusion"):
            parser.error("TASK must be 'tumor' or 'perfusion'")
        action_predict(task, file_path, raw_path)

    elif args.health:
        action_health()

    else:
        # Default: start server
        action_serve()


if __name__ == "__main__":
    main()