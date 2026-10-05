"""Verify server-side model bundles before production startup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = {
    "random_forest": ("model.joblib",),
    "xgboost": ("model.json",),
    "lstm": ("model.pt",),
}
COMMON = ("metadata.json", "metrics.json", "feature_names.json", "label_encoder.joblib", "preprocessor.joblib")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ml/artifacts", type=Path)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    bundles = []
    for meta in sorted(root.glob("*/*/*/metadata.json")):
        directory = meta.parent
        data = json.loads(meta.read_text())
        model = data.get("model")
        missing = [name for name in COMMON if not (directory / name).exists()]
        missing += [name for name in REQUIRED.get(model, ()) if not (directory / name).exists()]
        if missing:
            print(f"INVALID {directory}: missing {', '.join(missing)}")
            return 1
        bundles.append((directory, model))
    if not bundles:
        print(f"NO_ARTIFACTS {root}")
        return 1
    if args.require_all:
        present = {model for _, model in bundles}
        missing_models = set(REQUIRED) - present
        if missing_models:
            print(f"MISSING_MODELS {', '.join(sorted(missing_models))}")
            return 1
    for directory, model in bundles:
        print(f"OK {model}: {directory}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
