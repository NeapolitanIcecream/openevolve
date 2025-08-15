"""
Service for handling file discovery and filtering based on ignore files.
"""

import os
from typing import List

# The user needs to install this dependency: pip install pathspec
import pathspec

from .config import Config


class FileService:
    """
    Manages file discovery and filtering.
    """

    def __init__(self, config: Config):
        self.config = config
        self.gitignore_spec = self._load_spec_from_file(
            os.path.join(self.config.root_dir, ".gitignore")
        )
        # Gemini ignore is deprecated in this project; keep an empty spec for compatibility
        self.geminiignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", [])

    def _load_spec_from_file(self, file_path: str) -> pathspec.PathSpec:
        """Loads a .gitignore-style file and returns a PathSpec object."""
        patterns = []
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                patterns = f.read().splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    def get_gemini_ignore_patterns(self) -> List[str]:
        """Deprecated: returns an empty list (gemini ignore disabled)."""
        return []

    def should_gemini_ignore_file(self, file_path: str) -> bool:
        """Checks if a file should be ignored based on .geminiignore rules."""
        relative_path = os.path.relpath(file_path, self.config.root_dir)
        return self.geminiignore_spec.match_file(relative_path)

    def filter_files(
        self, files: List[str], respect_git_ignore: bool, respect_gemini_ignore: bool
    ) -> List[str]:
        """Filters a list of files based on the ignore rules."""
        # respect_gemini_ignore is ignored (deprecated)
        if not respect_git_ignore:
            return files

        filtered_files = []
        for file_path in files:
            # The pathspec library works with relative paths
            relative_path = os.path.relpath(file_path, self.config.root_dir)
            is_git_ignored = respect_git_ignore and self.gitignore_spec.match_file(relative_path)

            if not is_git_ignored:
                filtered_files.append(file_path)

        return filtered_files


file_service = FileService(Config(root_dir=os.getcwd()))
