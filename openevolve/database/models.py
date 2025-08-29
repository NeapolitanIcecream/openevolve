from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, TypedDict


class DiversityCacheEntry(TypedDict):
    value: float
    timestamp: float


class FeatureStats(TypedDict):
    min: float
    max: float


@dataclass
class Program:
    """Represents a program in the database"""

    # Program identification
    id: str
    # Commit identifier (no longer store full code text)
    commit_hash: str
    # Added: commit information and its diff
    prompt_diff: Optional[str] = None  # Cleaned diff with filenames, for prompts
    hash_diff: Optional[str] = None    # Fully cleaned diff, used for MinHash
    minhash_signature: List[int] = field(default_factory=list)
    language: str = "python"

    # Evolution information
    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    iteration_found: int = 0  # Track which iteration this program was found

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Derived features
    complexity: float = 0.0
    diversity: float = 0.0

    # Metadata
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation"""
        # Get the valid field names for the Program dataclass
        valid_fields = {f.name for f in fields(cls)}

        # Filter the data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)


