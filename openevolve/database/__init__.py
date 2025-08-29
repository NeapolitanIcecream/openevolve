"""Database package exposing stable public API.

Keep backward compatibility for:
    from openevolve.database import Program, ProgramDatabase
"""

from .models import Program
from .service import ProgramDatabase

__all__ = [
    "Program",
    "ProgramDatabase",
]


