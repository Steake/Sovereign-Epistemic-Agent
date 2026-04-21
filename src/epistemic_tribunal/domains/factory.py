from __future__ import annotations

from epistemic_tribunal.tribunal_types import TaskDomain
from epistemic_tribunal.domains.base import DomainAdapter
from epistemic_tribunal.domains.arc_grid import ArcGridAdapter
from epistemic_tribunal.domains.gsm8k_math import Gsm8kMathAdapter

_ADAPTERS: dict[TaskDomain, DomainAdapter] = {
    TaskDomain.ARC_LIKE: ArcGridAdapter(),
    TaskDomain.GSM8K_MATH: Gsm8kMathAdapter(),
}

def get_adapter(domain: TaskDomain) -> DomainAdapter:
    """Return the correct DomainAdapter for a given TaskDomain."""
    adapter = _ADAPTERS.get(domain)
    if adapter is None:
        # Fallback to ARC_LIKE for unknown generic domains for backward compatibility
        return _ADAPTERS[TaskDomain.ARC_LIKE]
    return adapter
