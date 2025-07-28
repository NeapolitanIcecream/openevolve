"""
Defines configuration classes for tools.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class FileFilteringOptions:
    """Options for file filtering."""

    respect_git_ignore: bool = True
    respect_gemini_ignore: bool = True


@dataclass
class Config:
    """
    Configuration for the tool registry and its tools.
    """

    root_dir: str
    file_filtering_options: FileFilteringOptions = field(default_factory=FileFilteringOptions)
    # Placeholder for other config properties if needed
    other_config: Dict[str, Any] = field(default_factory=dict)


# Default options to be used if not specified.
DEFAULT_FILE_FILTERING_OPTIONS = FileFilteringOptions()
