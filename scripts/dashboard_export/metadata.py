"""System metadata extraction — model architecture, training config, trading params.

Reads project source code (architecture.py), YAML configs (task_rendered.yaml,
trading.yaml), and assembles structured metadata for the dashboard.
All secrets and sensitive fields are sanitized before export.
"""

from __future__ import annotations

from typing import Any

from scripts.dashboard_export.constants import REPO_ROOT, log


def extract_system_info() -> dict[str, Any]:
    """Extract model architecture, training config, and project metadata."""
    info: dict[str, Any] = {
        "project": {
            "name": "FullStackAutoQuant",
            "description": "End-to-End Deep Learning Quantitative Trading System",
            "author": "Zizhao Huang",
            "license": "CC-BY-NC-SA-4.0",
            "repository": "https://github.com/Zizhao-HUANG/FullStackAutoQuant",
        },
        "model": {},
        "training": {},
        "data_pipeline": {},
    }

    # Model architecture info from architecture.py
    try:
        from fullstackautoquant.model.architecture import Net
        net = Net(num_features=22, num_timesteps=72)
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

        info["model"] = {
            "name": getattr(net, "model_name", "TCN_LocalAttn_GRU"),
            "type": getattr(net, "model_type", "TimeSeries"),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_features": 22,
            "input_timesteps": 72,
            "d_model": getattr(net, "d_model", 96),
            "gru_hidden_size": getattr(net, "hidden_size", 48),
            "attention_heads": getattr(net, "nhead", 6),
            "attention_head_dim": getattr(net, "head_dim", 16),
            "dropout": getattr(net, "dropout", 0.1),
            "window_size": getattr(net, "window_size", 16),
            "window_stride": getattr(net, "window_stride", 8),
            "architecture_components": [
                "Linear Embedding (22→96)",
                "3× Residual TCN Blocks (causal convolution, GELU, dilation)",
                "Post-TCN Projection (64→96)",
                "Overlapping Local Self-Attention (causal mask, window=16, stride=8)",
                "GRU Head (96→48, unidirectional)",
                "Output MLP (48→32→1, GELU + Dropout)",
            ],
        }

        # Training hyperparameters
        training_hp = getattr(net, "training_hyperparameters", {})
        if training_hp:
            info["training"] = {
                "n_epochs": training_hp.get("n_epochs"),
                "learning_rate": training_hp.get("lr"),
                "batch_size": training_hp.get("batch_size"),
                "weight_decay": training_hp.get("weight_decay"),
                "precision": training_hp.get("precision"),
                "loss_function": training_hp.get("loss_fn"),
                "optimizer": training_hp.get("optimizer"),
                "lr_scheduler": training_hp.get("lr_scheduler"),
                "gradient_clip_norm": training_hp.get("gradient_clip_norm"),
                "seed": training_hp.get("seed"),
                "step_len": training_hp.get("step_len"),
                "rank_loss_margin": training_hp.get("rank_loss_margin"),
                "rank_mse_blend_alpha": training_hp.get("rank_mse_blend_alpha"),
                "turnover_regularization": training_hp.get("turnover_regularization_lambda"),
            }
    except Exception as exc:
        log("WARN", f"Cannot extract model info: {exc}")

    # Data pipeline info from task_rendered.yaml
    _extract_data_pipeline_info(info)

    # MC Dropout inference config
    info["inference"] = {
        "mc_dropout_samples": 16,
        "confidence_metric": "1 / (1 + std(MC predictions))",
        "normalization": "Cached RobustZScoreNorm (norm_params.pkl)",
        "device": "CPU",
    }

    return info


def _extract_data_pipeline_info(info: dict[str, Any]) -> None:
    """Extract data pipeline and backtest config from task_rendered.yaml."""
    task_yaml = REPO_ROOT / "configs" / "task_rendered.yaml"
    if not task_yaml.exists():
        return

    try:
        import yaml
        with open(task_yaml, encoding="utf-8") as f:
            task_cfg = yaml.safe_load(f)

        handler = task_cfg.get("dataset", {}).get("kwargs", {}).get("handler", {})
        handler_kwargs = handler.get("kwargs", {})
        segments = task_cfg.get("dataset", {}).get("kwargs", {}).get("segments", {})

        # Extract feature names from config
        dl_cfg = handler_kwargs.get("data_loader", {}).get("kwargs", {})
        loaders = dl_cfg.get("dataloader_l", [])
        feature_names: list[str] = []
        if loaders and len(loaders) > 0:
            alpha_cfg = loaders[0].get("kwargs", {}).get("config", {})
            if isinstance(alpha_cfg, dict) and "feature" in alpha_cfg:
                features = alpha_cfg["feature"]
                if len(features) >= 2:
                    feature_names = features[1]  # Second list is names

        info["data_pipeline"] = {
            "universe": handler_kwargs.get("instruments", "csi300"),
            "start_time": handler_kwargs.get("start_time"),
            "end_time": handler_kwargs.get("end_time"),
            "alpha_features": feature_names,
            "num_alpha_features": len(feature_names),
            "custom_factors": 2,  # combined_factors_df.parquet adds 2 factors
            "total_features": len(feature_names) + 2,
            "normalization": "RobustZScoreNorm (fit: 2005-01-04 to 2021-12-31, clip_outlier=True)",
            "label": "5-day asymmetric return × volume ratio",
            "segments": {
                "train": segments.get("train", []),
                "valid": segments.get("valid", []),
                "test": segments.get("test", []),
            },
            "step_len": task_cfg.get("dataset", {}).get("kwargs", {}).get("step_len", 72),
        }

        # Backtest config
        records = task_cfg.get("record", [])
        port_record = next((r for r in records if r.get("class") == "PortAnaRecord"), None)
        if port_record:
            bt_cfg = port_record.get("kwargs", {}).get("config", {})
            bt = bt_cfg.get("backtest", {})
            strat = bt_cfg.get("strategy", {})
            info["backtest_config"] = {
                "benchmark": bt.get("benchmark"),
                "initial_capital": bt.get("account"),
                "start_time": bt.get("start_time"),
                "end_time": bt.get("end_time"),
                "transaction_costs": {
                    "open_cost": bt.get("exchange_kwargs", {}).get("open_cost"),
                    "close_cost": bt.get("exchange_kwargs", {}).get("close_cost"),
                    "min_cost": bt.get("exchange_kwargs", {}).get("min_cost"),
                },
                "strategy": {
                    "class": strat.get("class"),
                    "topk": strat.get("kwargs", {}).get("topk"),
                    "n_drop": strat.get("kwargs", {}).get("n_drop"),
                },
            }
    except Exception as exc:
        log("WARN", f"Cannot parse task_rendered.yaml: {exc}")


def extract_trading_config() -> dict[str, Any]:
    """Extract sanitized trading configuration (no secrets)."""
    trading_yaml = REPO_ROOT / "configs" / "trading.yaml"
    if not trading_yaml.exists():
        return {}

    try:
        import yaml
        with open(trading_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        log("WARN", f"Cannot parse trading.yaml: {exc}")
        return {}

    # Sanitize: keep only safe sections
    sanitized: dict[str, Any] = {}
    for section in ["portfolio", "weights", "order", "risk", "rebalance_trigger"]:
        if section in cfg:
            sanitized[section] = cfg[section]

    # Remove secret-containing sections (gm, capital)
    sanitized["capital"] = {"note": "Real capital amount redacted for privacy"}

    return sanitized
