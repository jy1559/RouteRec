"""
Structured file-based logging for RouteRecBase experiments.

Creates per-run output directories under ``outputs/RouteRec/`` containing:
  - ``config.json``       — full experiment configuration
  - ``epoch_metrics.csv`` — per-epoch train/valid metrics
  - ``summary.json``      — final results (best valid, test, timing)

When ``debug_logging=True``, additional analysis files are written:
  - ``expert_weights.csv``    — per-epoch, per-stage, per-expert gating statistics
  - ``expert_performance.csv``— expert weights for correct vs incorrect predictions
  - ``feature_bias.csv``      — feature bucket to expert bias analysis

Usage in training loop::

    from models.RouteRecBase.run_logger import RunLogger

    run_log = RunLogger(run_name="AMA_FME_eS_0130", config=cfg_dict)
    run_log.log_epoch(epoch=0, train_loss=24.5, valid_result={...})
    run_log.log_expert_weights(epoch=0, moe_summary={...})
    run_log.log_final(best_valid={...}, test_result={...}, elapsed=120.0)
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default output root (relative to experiments/ working directory)
_DEFAULT_OUTPUT_ROOT = "run/artifacts/logging"


class RunLogger:
    """Structured per-run file logger for RouteRecBase experiments.

    Parameters
    ----------
    run_name : str
        Unique run identifier (e.g., "AMA_FME_eS_01301530").
    config : dict
        Full experiment configuration to save.
    output_root : str or Path
        Root directory for all RouteRec run outputs.
        Defaults to ``outputs/RouteRec/`` under the current working directory.
    debug_logging : bool
        If False, only metric files are written (epoch_metrics + summary).
        If True, expert/bucket analysis CSV files are also written.
    """

    def __init__(
        self,
        run_name: str,
        config: Optional[Dict[str, Any]] = None,
        output_root: Optional[str] = None,
        debug_logging: bool = False,
    ):
        self.run_name = run_name
        self.debug_logging = bool(debug_logging)
        self.config = dict(config or {})
        root = Path(output_root or _DEFAULT_OUTPUT_ROOT)

        model_scope = _derive_model_scope(self.config)
        axis = _sanitize_token(str(self.config.get("run_axis", "")))
        dataset = _sanitize_token(str(self.config.get("dataset", "")))
        phase = _sanitize_token(str(self.config.get("routerec_phase", self.config.get("phase", "P0"))))
        run_id = _sanitize_token(str(self.config.get("routerec_run_id", "")))
        arch = _sanitize_token(str(self.config.get("routerec_architecture_id", "")))
        hparam = _sanitize_token(str(self.config.get("routerec_hparam_id", self.config.get("phase_hparam_id", ""))))
        layout = str(self.config.get("routerec_logging_layout", "")).strip().lower()
        if layout == "axis_dataset_arch_hparam" and model_scope and axis and dataset and arch and hparam and run_id:
            self.run_dir = root / model_scope / axis / dataset / arch / hparam / run_id
        elif model_scope and dataset and phase and run_id:
            self.run_dir = root / model_scope / dataset / phase / run_id
        else:
            self.run_dir = root / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self._config_path = self.run_dir / "config.json"
        self._run_meta_path = self.run_dir / "run_meta.json"
        self._epoch_csv = self.run_dir / "metrics_epoch.csv"
        self._expert_csv = self.run_dir / "expert_weights.csv"
        self._perf_csv = self.run_dir / "expert_performance.csv"
        self._bias_csv = self.run_dir / "feature_bias.csv"
        self._summary_path = self.run_dir / "metrics_summary.json"
        self._special_metrics_path = self.run_dir / "special_metrics.json"
        self._router_diag_path = self.run_dir / "router_diag.json"

        # Write config
        if config is not None:
            self._write_config(config)
        self._write_run_meta(status="started")

        # Init CSV files (write headers)
        self._init_epoch_csv()
        if self.debug_logging:
            self._init_expert_csv()
            self._init_perf_csv()
            self._init_bias_csv()

        # In-memory accumulator for epoch data (used in summary)
        self._epoch_data: List[Dict] = []

        logger.info(
            f"RunLogger: saving to {self.run_dir} (debug_logging={self.debug_logging})"
        )

    def _write_run_meta(self, *, status: str, extra: Optional[Dict[str, Any]] = None):
        payload = {
            "run_name": self.run_name,
            "run_id": str(self.config.get("routerec_run_id", "")),
            "model_scope": _derive_model_scope(self.config),
            "model": str(self.config.get("model", "")),
            "dataset": str(self.config.get("dataset", "")),
            "phase": str(self.config.get("routerec_phase", self.config.get("phase", ""))),
            "run_phase": str(self.config.get("run_phase", "")),
            "seed": self.config.get("seed", None),
            "status": status,
            "updated_at": datetime.now().isoformat(),
        }
        if extra:
            payload.update(_make_serializable(extra))
        with open(self._run_meta_path, "w") as f:
            json.dump(_make_serializable(payload), f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _write_config(self, config: Dict[str, Any]):
        """Write experiment config to JSON."""
        # Make config JSON-serializable (convert non-serializable values)
        safe = _make_serializable(config)
        safe["_run_name"] = self.run_name
        safe["_created_at"] = datetime.now().isoformat()
        with open(self._config_path, "w") as f:
            json.dump(safe, f, indent=2, ensure_ascii=False)

    def update_config(self, config: Dict[str, Any]):
        """Rewrite config.json (used when runtime-resolved fields become available)."""
        self._write_config(config)

    # ------------------------------------------------------------------
    # Epoch metrics CSV
    # ------------------------------------------------------------------

    def _init_epoch_csv(self):
        """Create epoch_metrics.csv with header."""
        if not self._epoch_csv.exists():
            with open(self._epoch_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "train_loss",
                    "hit@5", "hit@10", "hit@20",
                    "ndcg@5", "ndcg@10", "ndcg@20",
                    "mrr@5", "mrr@10", "mrr@20",
                    "did_eval",
                    "epoch_time_sec",
                ])

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        valid_result: Dict[str, float],
        did_eval: bool = True,
        epoch_time: float = 0.0,
    ):
        """Append one row to epoch_metrics.csv.

        Parameters
        ----------
        epoch : int
            0-indexed epoch number.
        train_loss : float
            Training loss for this epoch.
        valid_result : dict
            Validation metrics dict (keys like 'hit@5', 'mrr@20', etc.).
        epoch_time : float
            Wall-clock time for this epoch in seconds.
        """
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "hit@5": valid_result.get("hit@5", 0),
            "hit@10": valid_result.get("hit@10", 0),
            "hit@20": valid_result.get("hit@20", 0),
            "ndcg@5": valid_result.get("ndcg@5", 0),
            "ndcg@10": valid_result.get("ndcg@10", 0),
            "ndcg@20": valid_result.get("ndcg@20", 0),
            "mrr@5": valid_result.get("mrr@5", 0),
            "mrr@10": valid_result.get("mrr@10", 0),
            "mrr@20": valid_result.get("mrr@20", 0),
            "did_eval": int(bool(did_eval)),
            "epoch_time_sec": epoch_time,
        }
        self._epoch_data.append(row)

        with open(self._epoch_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                row["epoch"], f"{row['train_loss']:.6f}",
                f"{row['hit@5']:.6f}", f"{row['hit@10']:.6f}", f"{row['hit@20']:.6f}",
                f"{row['ndcg@5']:.6f}", f"{row['ndcg@10']:.6f}", f"{row['ndcg@20']:.6f}",
                f"{row['mrr@5']:.6f}", f"{row['mrr@10']:.6f}", f"{row['mrr@20']:.6f}",
                f"{row['did_eval']}",
                f"{row['epoch_time_sec']:.2f}",
            ])

    # ------------------------------------------------------------------
    # Expert weight CSV
    # ------------------------------------------------------------------

    def _init_expert_csv(self):
        """Create expert_weights.csv with header."""
        if not self._expert_csv.exists():
            with open(self._expert_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "stage", "expert_name",
                    "mean_weight", "std_weight", "balance_score",
                ])

    def log_expert_weights(self, epoch: int, moe_summary: Dict):
        """Append expert weight rows from an MoELogger summary dict.

        Parameters
        ----------
        epoch : int
            0-indexed epoch number.
        moe_summary : dict
            Output of ``model.get_epoch_log_summary()``.
        """
        if not self.debug_logging:
            return
        if not moe_summary or moe_summary.get("n_batches", 0) == 0:
            return

        with open(self._expert_csv, "a", newline="") as f:
            writer = csv.writer(f)
            for stage_name, s in moe_summary.get("stages", {}).items():
                names = s.get("expert_names", [])
                means = s.get("mean_weights", [])
                stds = s.get("std_weights", [])
                balance = s.get("balance_score", 0.0)
                for name, m, sd in zip(names, means, stds):
                    writer.writerow([
                        epoch, stage_name, name,
                        f"{m:.6f}", f"{sd:.6f}", f"{balance:.6f}",
                    ])

    # ------------------------------------------------------------------
    # Expert performance CSV (정답/오답별 expert 가중치)
    # ------------------------------------------------------------------

    def _init_perf_csv(self):
        """Create expert_performance.csv with header."""
        if not self._perf_csv.exists():
            with open(self._perf_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "stage", "expert_name",
                    "correct_weight", "incorrect_weight", "weight_diff",
                    "n_correct", "n_incorrect",
                ])

    # ------------------------------------------------------------------
    # Feature bias CSV (feature 구간별 expert 선택 분포)
    # ------------------------------------------------------------------

    def _init_bias_csv(self):
        """Create feature_bias.csv with header."""
        if not self._bias_csv.exists():
            with open(self._bias_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "stage", "feature", "bin", "bin_label",
                    "expert_name", "mean_weight", "count",
                ])

    def log_analysis(self, epoch: int, analysis: Dict):
        """Write expert analysis results to CSV files.

        Parameters
        ----------
        epoch : int
            0-indexed epoch number.
        analysis : dict
            Output of ``ExpertAnalysisLogger.get_and_reset()``.
        """
        if not self.debug_logging:
            return
        if not analysis or analysis.get("n_sampled", 0) == 0:
            return

        # --- Performance CSV ---
        perf = analysis.get("performance", {})
        if perf:
            with open(self._perf_csv, "a", newline="") as f:
                writer = csv.writer(f)
                for stage_name, data in perf.items():
                    names = data.get("expert_names", [])
                    corr = data.get("correct_mean_weights", [])
                    incorr = data.get("incorrect_mean_weights", [])
                    diff = data.get("weight_diff", [])
                    nc = data.get("n_correct", 0)
                    ni = data.get("n_incorrect", 0)
                    for n, c, i, d in zip(names, corr, incorr, diff):
                        writer.writerow([
                            epoch, stage_name, n,
                            f"{c:.6f}", f"{i:.6f}", f"{d:+.6f}",
                            nc, ni,
                        ])

        # --- Feature bias CSV ---
        fbias = analysis.get("feature_bias", {})
        if fbias:
            with open(self._bias_csv, "a", newline="") as f:
                writer = csv.writer(f)
                for stage_name, features in fbias.items():
                    # Need expert names for this stage
                    stage_bins = next(iter(features.values()), {}).get("bins", [])
                    inferred_k = 0
                    if stage_bins:
                        inferred_k = len(stage_bins[0].get("mean_weights", []))
                    stage_expert_names = perf.get(stage_name, {}).get(
                        "expert_names",
                        [f"E{i}" for i in range(inferred_k)],
                    )
                    for feat_name, data in features.items():
                        for bd in data.get("bins", []):
                            bin_idx = bd["bin"]
                            label = bd.get("label", f"Q{bin_idx+1}")
                            weights = bd.get("mean_weights", [])
                            count = bd.get("count", 0)
                            for ename, w in zip(stage_expert_names, weights):
                                writer.writerow([
                                    epoch, stage_name, feat_name,
                                    bin_idx, label, ename,
                                    f"{w:.6f}", count,
                                ])

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    def log_final(
        self,
        best_valid_result: Dict[str, float],
        test_result: Dict[str, float],
        elapsed_time: float,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Write final summary JSON.

        Parameters
        ----------
        best_valid_result : dict
            Best validation metrics.
        test_result : dict
            Test set metrics.
        elapsed_time : float
            Total training time in seconds.
        extra : dict, optional
            Additional info to include in summary.
        """
        summary = {
            "run_name": self.run_name,
            "finished_at": datetime.now().isoformat(),
            "elapsed_time_sec": elapsed_time,
            "elapsed_time_min": elapsed_time / 60,
            "n_epochs_run": len(self._epoch_data),
            "best_valid": _make_serializable(best_valid_result),
            "test": _make_serializable(test_result),
        }
        if extra:
            summary["extra"] = _make_serializable(extra)

        with open(self._summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self._write_run_meta(
            status="finished",
            extra={
                "elapsed_time_sec": elapsed_time,
                "n_epochs_run": len(self._epoch_data),
                "best_valid_mrr20": best_valid_result.get("mrr@20", None),
                "test_mrr20": test_result.get("mrr@20", None),
            },
        )

        logger.info(f"RunLogger: summary saved to {self._summary_path}")

    def log_special_metrics(
        self,
        *,
        valid_special_metrics: Optional[Dict[str, Any]],
        test_special_metrics: Optional[Dict[str, Any]],
    ):
        """Write aggregated special slice metrics to one run-level JSON file."""
        if valid_special_metrics is None and test_special_metrics is None:
            return

        config_snapshot = {}
        if isinstance(valid_special_metrics, dict):
            config_snapshot = dict(valid_special_metrics.get("config_snapshot", {}))
        if not config_snapshot and isinstance(test_special_metrics, dict):
            config_snapshot = dict(test_special_metrics.get("config_snapshot", {}))

        payload = {
            "valid": (valid_special_metrics or {}).get("overall", {}),
            "valid_overall_seen_target": (valid_special_metrics or {}).get("overall_seen_target", {}),
            "valid_overall_unseen_target": (valid_special_metrics or {}).get("overall_unseen_target", {}),
            "test": (test_special_metrics or {}).get("overall", {}),
            "test_overall_seen_target": (test_special_metrics or {}).get("overall_seen_target", {}),
            "test_overall_unseen_target": (test_special_metrics or {}).get("overall_unseen_target", {}),
            "slice_metrics": {
                "valid": (valid_special_metrics or {}).get("slices", {}),
                "test": (test_special_metrics or {}).get("slices", {}),
            },
            "counts": {
                "valid": (valid_special_metrics or {}).get("counts", {}),
                "test": (test_special_metrics or {}).get("counts", {}),
            },
            "full_payload": {
                "valid": _make_serializable(valid_special_metrics or {}),
                "test": _make_serializable(test_special_metrics or {}),
            },
            "selection_rule_default": "overall_seen_target",
            "config_snapshot": _make_serializable(config_snapshot),
        }

        with open(self._special_metrics_path, "w") as f:
            json.dump(_make_serializable(payload), f, indent=2, ensure_ascii=False)

        logger.info(f"RunLogger: special metrics saved to {self._special_metrics_path}")

    def log_router_diagnostics(
        self,
        *,
        valid_diag: Optional[Dict[str, Any]],
        test_diag: Optional[Dict[str, Any]],
    ):
        if valid_diag is None and test_diag is None:
            return
        payload = {
            "valid": _make_serializable(valid_diag or {}),
            "test": _make_serializable(test_diag or {}),
            "created_at": datetime.now().isoformat(),
        }
        with open(self._router_diag_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"RunLogger: router diagnostics saved to {self._router_diag_path}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def output_path(self) -> Path:
        """Return the run output directory path."""
        return self.run_dir


def _make_serializable(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, "item"):  # numpy/torch scalar
        return obj.item()
    elif hasattr(obj, "tolist"):  # numpy/torch array
        return obj.tolist()
    else:
        return str(obj)


def _sanitize_token(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    out = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token


def _derive_model_scope(config: Dict[str, Any]) -> str:
    key = str((config or {}).get("routerec_logging_model_scope", "")).strip()
    if key:
        return _sanitize_token(key)
    model = str((config or {}).get("model", "")).lower()
    if "routerec_n3" in model or "routerec" in model:
        return "routerec_n3"
    if "routerec" in model:
        return "routerec"
    return ""
