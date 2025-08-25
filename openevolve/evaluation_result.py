"""
Evaluation result structures for OpenEvolve
"""

import json
from dataclasses import dataclass
from typing import Dict


@dataclass
class EvaluationResult:
    """
    Result of program evaluation containing metrics only.
    """

    metrics: Dict[str, float]

    @classmethod
    def from_dict(cls, metrics: Dict[str, float]) -> "EvaluationResult":
        """Auto-wrap dict returns for backward compatibility"""
        return cls(metrics=metrics)

    def to_dict(self) -> Dict[str, float]:
        """Backward compatibility - return just metrics"""
        return self.metrics

    
