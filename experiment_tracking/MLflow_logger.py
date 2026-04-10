"""
mlflow_logger.py
────────────────
Logging helpers for the Momento compatibility pipeline.

Structure
─────────
  experiment: momento_compatibility
    └── parent run  (one per user-pair × passage)
          ├── params : model, temperature, prompt_version, book, passage, user_a, user_b
          ├── metrics: confidence, think_R/C/D, feel_R/C/D, match_count
          ├── tags   : dominant_think, dominant_feel, route
          └── child run — decomposition_A
          └── child run — decomposition_B
                ├── params : user_id, passage_id, book_id
                ├── metrics: subclaim_count, weight_entropy, min/max weight
                └── tags   : emotional_modes (comma-separated)
"""

import os
import json
import math
import tempfile
import mlflow # type: ignore
import yaml # type: ignore
from pathlib import Path


# ── Load config ───────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_mlflow(config: dict | None = None) -> str:
    """
    Point MLflow at the local mlruns/ folder and create the experiment
    if it doesn't exist yet. Returns the experiment_id.
    """
    cfg = config or load_config()
    mlflow_cfg = cfg["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    experiment = mlflow.set_experiment(mlflow_cfg["experiment_name"])
    print(f"[MLflow] experiment '{mlflow_cfg['experiment_name']}' "
          f"(id={experiment.experiment_id})")
    return experiment.experiment_id


# ── Decomposition child run ───────────────────────────────────────────────────

def log_decomposition_run(decomp: dict, reader_label: str, config: dict) -> None:
    """
    Log one decomposition as a *child* MLflow run.
    Must be called inside an active parent run context.

    Args:
        decomp       : output dict from run_decomposer()
        reader_label : "reader_a" or "reader_b"
        config       : full config dict
    """
    model_cfg = config["model"]
    subclaims = decomp.get("subclaims", [])
    weights   = [sc["weight"] for sc in subclaims]
    modes     = [sc["emotional_mode"] for sc in subclaims]

    # Shannon entropy of weight distribution — higher = more evenly split
    entropy = 0.0
    for w in weights:
        if w > 0:
            entropy -= w * math.log2(w)

    with mlflow.start_run(
        run_name=f"decomp_{reader_label}_{decomp.get('user_id', 'unknown')}",
        nested=True,
    ):
        # params
        mlflow.log_params({
            "user_id":        decomp.get("user_id"),
            "passage_id":     decomp.get("passage_id"),
            "book_id":        decomp.get("book_id"),
            "model_name":     model_cfg["name"],
            "temperature":    model_cfg["temperature"],
            "prompt_version": model_cfg["prompt_version"],
            "reader_label":   reader_label,
        })

        # metrics
        mlflow.log_metrics({
            "subclaim_count":  len(subclaims),
            "weight_entropy":  round(entropy, 4),
            "weight_min":      round(min(weights), 4) if weights else 0.0,
            "weight_max":      round(max(weights), 4) if weights else 0.0,
        })

        # tags
        mlflow.set_tags({
            "emotional_modes": ", ".join(sorted(set(modes))),
            "dominant_mode":   max(set(modes), key=modes.count) if modes else "none",
        })

        # artifact — full decomposition JSON
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix=f"decomp_{reader_label}_"
        ) as tmp:
            json.dump(decomp, tmp, indent=2)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, artifact_path="decompositions")
        os.unlink(tmp_path)


# ── Compatibility parent run ──────────────────────────────────────────────────

def log_compatibility_run(
    result: dict,
    decomp_a: dict,
    decomp_b: dict,
    config: dict,
) -> str:
    """
    Log a full compatibility result as a parent MLflow run, with two
    nested child runs for the decompositions.

    Args:
        result   : output dict from aggregate() + metadata
        decomp_a : decomposition for user_a
        decomp_b : decomposition for user_b
        config   : full config dict

    Returns:
        mlflow run_id of the parent run
    """
    model_cfg  = config["model"]
    mlflow_cfg = config["mlflow"]

    think = result.get("think", {})
    feel  = result.get("feel",  {})

    run_name = (
        f"{result.get('user_a','?')} × {result.get('user_b','?')} "
        f"| {result.get('book_id','?')} / {result.get('passage_id','?')}"
    )

    with mlflow.start_run(
        run_name=run_name,
        tags=mlflow_cfg.get("run_tags", {}),
    ) as parent_run:

        # ── params ────────────────────────────────────────────────────────────
        mlflow.log_params({
            "user_a":         result.get("user_a"),
            "user_b":         result.get("user_b"),
            "book_id":        result.get("book_id"),
            "passage_id":     result.get("passage_id"),
            "model_name":     model_cfg["name"],
            "temperature":    model_cfg["temperature"],
            "prompt_version": model_cfg["prompt_version"],
        })

        # ── metrics ───────────────────────────────────────────────────────────
        mlflow.log_metrics({
            "confidence":   result.get("confidence", 0.0),
            "match_count":  result.get("match_count", 0),
            "think_R":      think.get("R", 0),
            "think_C":      think.get("C", 0),
            "think_D":      think.get("D", 0),
            "feel_R":       feel.get("R",  0),
            "feel_C":       feel.get("C",  0),
            "feel_D":       feel.get("D",  0),
        })

        # ── tags ──────────────────────────────────────────────────────────────
        mlflow.set_tags({
            "dominant_think": result.get("dominant_think", "unknown"),
            "dominant_feel":  result.get("dominant_feel",  "unknown"),
            "route":          result.get("route", "display"),
            "verdict":        str(result.get("verdict")),
        })

        # ── artifact — full result JSON ────────────────────────────────────────
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="compat_result_"
        ) as tmp:
            json.dump(result, tmp, indent=2)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, artifact_path="compatibility_results")
        os.unlink(tmp_path)

        # ── nested decomposition runs ─────────────────────────────────────────
        log_decomposition_run(decomp_a, "reader_a", config)
        log_decomposition_run(decomp_b, "reader_b", config)

        return parent_run.info.run_id


# ── Batch artifact snapshot ───────────────────────────────────────────────────

def log_batch_artifacts(
    decompositions_path: str,
    compat_runs_path: str,
) -> None:
    """
    Attach the raw decompositions.json and compatibility_runs.json
    as artifacts in a fresh standalone run after a full sweep.

    Args:
        decompositions_path : path to decompositions.json
        compat_runs_path    : path to compatibility_runs.json
    """
    with mlflow.start_run(run_name="__batch_snapshot__"):
        if os.path.exists(decompositions_path):
            mlflow.log_artifact(decompositions_path, artifact_path="snapshots")
        if os.path.exists(compat_runs_path):
            mlflow.log_artifact(compat_runs_path, artifact_path="snapshots")
        run_id = mlflow.active_run().info.run_id
    print(f"[MLflow] batch artifacts logged to run {run_id}")