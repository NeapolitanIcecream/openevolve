from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List

from openevolve.config import DatabaseConfig
from openevolve.database.models import FeatureStats


class FeatureStatsManager:
    """Manage feature statistics and scaling.

    This module extracts and encapsulates all logic related to per-feature
    sliding-window statistics and value scaling from the original database.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config

        # Feature scaling infrastructure
        self.feature_stats: Dict[str, FeatureStats] = {}
        self.feature_scaling_method: str = getattr(config, "feature_scaling_method", "minmax")

        # Sliding window + periodic recompute configuration
        self.feature_stats_window_size: int = int(
            getattr(config, "feature_stats_window_size", 5000)
        )
        self.feature_stats_recompute_interval: int = int(
            getattr(config, "feature_stats_recompute_interval", 500)
        )
        self.feature_stats_min_samples: int = int(
            getattr(config, "feature_stats_min_samples", 50)
        )
        # Rebuild trigger on removals (fraction of population removed in a single operation)
        self.feature_stats_rebuild_remove_ratio: float = float(
            getattr(config, "feature_stats_rebuild_remove_ratio", 0.1)
        )
        # Robust scaling quantiles (inclusive lower, upper)
        self.feature_stats_robust_low_q: float = float(
            getattr(config, "feature_stats_robust_low_q", 0.05)
        )
        self.feature_stats_robust_high_q: float = float(
            getattr(config, "feature_stats_robust_high_q", 0.95)
        )
        # Optional per-dimension scaling override
        self.feature_scaling_method_per_dim: Dict[str, str] = getattr(
            config, "feature_scaling_method_per_dim", {}
        ) or {}

        # In-memory sliding window buffers and caches (not persisted)
        self._feature_value_buffer: Dict[str, Deque[float]] = {}
        self._feature_updates_since_recompute: Dict[str, int] = {}
        # Cache structure per feature: { "min", "max", "q_low", "q_high", "mean", "std" }
        self._feature_stats_cache: Dict[str, Dict[str, float]] = {}

        # Runtime bookkeeping for scaling freeze
        self._feature_stats_updates_frozen: bool = False
        # Drift detection for triggering feature-map rebin at coordinator
        self._rebin_due_to_drift: bool = False

    # -------- Public API --------
    def freeze_after_warmup_if_needed(self, total_adds: int) -> bool:
        if bool(getattr(self.config, "feature_stats_freeze_enabled", False)) and not self._feature_stats_updates_frozen:
            try:
                warmup = int(getattr(self.config, "feature_stats_freeze_after_adds", 2000))
            except Exception:
                warmup = 2000
            if total_adds >= max(1, warmup):
                self._feature_stats_updates_frozen = True
                return True
        return False

    def update(self, feature_name: str, value: float) -> None:
        """Update statistics for a feature dimension (write-path only)."""
        if bool(getattr(self.config, "feature_stats_freeze_enabled", False)) and self._feature_stats_updates_frozen:
            return

        if feature_name not in self.feature_stats:
            self.feature_stats[feature_name] = {
                "min": value,
                "max": value,
            }

        stats = self.feature_stats[feature_name]
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)

        # Sliding window buffer for robust/modern scaling (not persisted)
        buf = self._feature_value_buffer.get(feature_name)
        if buf is None:
            buf = deque(maxlen=self.feature_stats_window_size)
            self._feature_value_buffer[feature_name] = buf
        buf.append(float(value))

        # Periodic recompute scheduling (purely periodic)
        self._feature_updates_since_recompute[feature_name] = (
            self._feature_updates_since_recompute.get(feature_name, 0) + 1
        )
        self._maybe_recompute_feature_stats(feature_name)

    def scale(self, feature_name: str, value: float) -> float:
        """Scale a feature value according to the configured scaling method, returns [0,1]."""
        method = self._get_scaling_method_for_dim(feature_name)

        # Prefer cached window stats when available
        cache = self._feature_stats_cache.get(feature_name, {})

        def _readonly_fallback() -> float:
            mode = getattr(self.config, "feature_readonly_fallback_mode", "use_feature_stats")
            if mode == "neutral_0_5":
                return 0.5
            if mode == "clip_0_1":
                return min(1.0, max(0.0, float(value)))
            if mode == "static_ranges":
                try:
                    static_ranges = getattr(self.config, "feature_readonly_static_minmax", {})
                    rng = static_ranges.get(feature_name)
                    if isinstance(rng, (list, tuple)) and len(rng) == 2:
                        min_v, max_v = float(rng[0]), float(rng[1])
                        if max_v <= min_v:
                            return 0.5
                        return min(1.0, max(0.0, (float(value) - min_v) / (max_v - min_v)))
                except Exception:
                    pass
                return 0.5
            # use_feature_stats (default)
            stats = self.feature_stats.get(feature_name)
            if stats is not None:
                min_v = stats.get("min")
                max_v = stats.get("max")
                if min_v is not None and max_v is not None and max_v != min_v:
                    return min(1.0, max(0.0, (float(value) - float(min_v)) / (float(max_v) - float(min_v))))
            return min(1.0, max(0.0, float(value)))

        if method == "robust":
            q_low = cache.get("q_low")
            q_high = cache.get("q_high")
            if q_low is None or q_high is None:
                return _readonly_fallback()
            if q_high <= q_low:
                return 0.5
            scaled = (float(value) - q_low) / (q_high - q_low)
            return min(1.0, max(0.0, scaled))

        if method == "minmax":
            min_val = cache.get("min")
            max_val = cache.get("max")
            if min_val is None or max_val is None:
                return _readonly_fallback()
            if max_val == min_val:
                return 0.5
            scaled = (float(value) - float(min_val)) / (float(max_val) - float(min_val))
            return min(1.0, max(0.0, scaled))

        if method == "percentile":
            values = list(self._feature_value_buffer.get(feature_name, []))
            if not values:
                return _readonly_fallback()
            count = sum(1 for v in values if v <= float(value))
            return count / len(values)

        if method == "zscore":
            mean = cache.get("mean")
            std = cache.get("std")
            if mean is None or std is None or std == 0:
                return _readonly_fallback()
            z = (float(value) - mean) / std
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        return _readonly_fallback()

    def calculate_complexity_bin(self, complexity: int, update_stats: bool, num_bins: int) -> int:
        if update_stats:
            self.update("complexity", float(complexity))
        scaled_value = self.scale("complexity", float(complexity))
        bin_idx = int(scaled_value * num_bins)
        return max(0, min(num_bins - 1, bin_idx))

    def calculate_diversity_bin(self, diversity: float, update_stats: bool, num_bins: int) -> int:
        if update_stats:
            self.update("diversity", diversity)
        scaled_value = self.scale("diversity", diversity)
        bin_idx = int(scaled_value * num_bins)
        return max(0, min(num_bins - 1, bin_idx))

    def rebuild_feature_stats_from_programs(self, programs: List[object], feature_dimensions: List[str]) -> None:
        # Reset buffers and counters
        self.feature_stats.clear()
        self._feature_value_buffer.clear()
        self._feature_updates_since_recompute.clear()
        self._feature_stats_cache.clear()

        if not programs:
            return

        from openevolve.utils.metrics_utils import safe_numeric_average

        def _append(feature_name: str, v: float) -> None:
            stats = self.feature_stats.get(feature_name)
            if stats is None:
                self.feature_stats[feature_name] = {"min": v, "max": v}
            else:
                stats["min"] = min(stats["min"], v)
                stats["max"] = max(stats["max"], v)
            buf = self._feature_value_buffer.get(feature_name)
            if buf is None:
                buf = deque(maxlen=self.feature_stats_window_size)
                self._feature_value_buffer[feature_name] = buf
            buf.append(float(v))

        for prog in programs:
            p = prog  # type: ignore
            # complexity
            if getattr(p, "hash_diff", None):
                comp = float(len(getattr(p, "hash_diff")))
            elif getattr(p, "prompt_diff", None):
                comp = float(len(getattr(p, "prompt_diff")))
            else:
                comp = 0.0
            _append("complexity", comp)

            # diversity
            try:
                div = float(getattr(p, "diversity")) if isinstance(getattr(p, "diversity"), (int, float)) else 0.0
            except Exception:
                div = 0.0
            _append("diversity", div)

            # aggregated score
            metrics = getattr(p, "metrics", {}) or {}
            if metrics:
                avg_score = safe_numeric_average(metrics)
                _append("score", float(avg_score))

            for dim in feature_dimensions:
                if dim not in ("complexity", "diversity", "score") and dim in metrics:
                    try:
                        _append(dim, float(metrics[dim]))
                    except Exception:
                        pass

        for feature_name in list(self._feature_value_buffer.keys()):
            try:
                self._recompute_feature_stats(feature_name)
                self._feature_updates_since_recompute[feature_name] = 0
            except Exception:
                continue

    # -------- Internals --------
    def _get_scaling_method_for_dim(self, feature_name: str) -> str:
        override = None
        try:
            override = self.feature_scaling_method_per_dim.get(feature_name)
        except Exception:
            override = None
        return override or self.feature_scaling_method

    def _maybe_recompute_feature_stats(self, feature_name: str) -> None:
        updates = int(self._feature_updates_since_recompute.get(feature_name, 0))
        buf = self._feature_value_buffer.get(feature_name)
        if not buf:
            return
        if len(buf) < self.feature_stats_min_samples:
            return
        if updates >= self.feature_stats_recompute_interval:
            # Save previous cache snapshot for drift detection
            prev = self._feature_stats_cache.get(feature_name, {}).copy()
            self._recompute_feature_stats(feature_name)
            self._feature_updates_since_recompute[feature_name] = 0
            # Detect drift to trigger rebin at coordinator if enabled
            if bool(getattr(self.config, "feature_map_rebin_enabled", False)):
                try:
                    drift_threshold = float(getattr(self.config, "feature_map_rebin_drift_threshold", 0.1))
                except Exception:
                    drift_threshold = 0.1
                curr = self._feature_stats_cache.get(feature_name, {})
                def _rel_change(old: float, new: float) -> float:
                    try:
                        denom = max(1e-9, abs(old))
                        return abs(new - old) / denom
                    except Exception:
                        return 0.0
                for k in ["min", "max", "q_low", "q_high"]:
                    if k in prev and k in curr:
                        if _rel_change(float(prev[k]), float(curr[k])) >= drift_threshold:
                            self._rebin_due_to_drift = True
                            break

    def _recompute_feature_stats(self, feature_name: str) -> None:
        buf = self._feature_value_buffer.get(feature_name)
        if not buf:
            return
        values = list(buf)
        if not values:
            return
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def _quantile(q: float) -> float:
            if n == 1:
                return sorted_vals[0]
            q = min(1.0, max(0.0, q))
            pos = q * (n - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return sorted_vals[lo]
            frac = pos - lo
            return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

        min_v = sorted_vals[0]
        max_v = sorted_vals[-1]
        low_q = _quantile(self.feature_stats_robust_low_q)
        high_q = _quantile(self.feature_stats_robust_high_q)
        mean_v = sum(values) / n
        var = sum((x - mean_v) * (x - mean_v) for x in values) / n
        std_v = math.sqrt(var) if var > 0 else 0.0

        self._feature_stats_cache[feature_name] = {
            "min": float(min_v),
            "max": float(max_v),
            "q_low": float(low_q),
            "q_high": float(high_q),
            "mean": float(mean_v),
            "std": float(std_v),
        }


