"""Conformal quantile and exact Poisson intervals."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import chi2


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal upper quantile: the ceil((n+1)(1-alpha))-th smallest score (inf if > n)."""
    s = np.sort(np.asarray(scores)[np.isfinite(scores)])
    k = math.ceil((s.size + 1) * (1 - alpha))
    return float("inf") if k > s.size else float(s[k - 1])


def poisson_rate_ci(count: int, hours: float, level: float = 0.95):
    """Garwood exact CI for a Poisson rate (events/hour). Assumes Poisson counts; with temporal
    dependence or between-subject heterogeneity it is optimistic (reported as such)."""
    a = 1 - level
    lo = 0.0 if count == 0 else chi2.ppf(a / 2, 2 * count) / 2
    hi = chi2.ppf(1 - a / 2, 2 * count + 2) / 2
    return count / hours, lo / hours, hi / hours
