from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from openevolve.config import DatabaseConfig
from openevolve.database.models import Program


class ProgramRepository:
    """In-memory repository with optional on-disk persistence."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.programs: Dict[str, Program] = {}

        # Persistence switch controlled by config.in_memory (False means enabled)
        self.persistence_enabled: bool = not getattr(config, "in_memory", True)

        if self.persistence_enabled:
            if not getattr(self.config, "db_path", None):
                try:
                    repo_path = getattr(self.config, "git_repo_path", ".") or "."
                    repo_abs = os.path.abspath(repo_path)
                    default_db_path = os.path.join(repo_abs, ".openevolve", "db")
                    self.config.db_path = default_db_path
                except Exception:
                    pass

    # CRUD
    def put(self, program: Program) -> None:
        self.programs[program.id] = program

    def get(self, program_id: str) -> Optional[Program]:
        return self.programs.get(program_id)

    def remove(self, program_id: str) -> None:
        if program_id in self.programs:
            del self.programs[program_id]

    # Persistence
    def save_all(self, path: Optional[str] = None, iteration: int = 0) -> None:
        if not self.persistence_enabled and path is None:
            return
        save_path = path or self.config.db_path
        if not save_path:
            return
        os.makedirs(save_path, exist_ok=True)
        for program in self.programs.values():
            self._save_program(program, save_path)

    def load_all(self, path: str) -> None:
        if not os.path.exists(path):
            return
        programs_dir = os.path.join(path, "programs")
        if os.path.exists(programs_dir):
            for program_file in os.listdir(programs_dir):
                if program_file.endswith(".json"):
                    program_path = os.path.join(programs_dir, program_file)
                    try:
                        with open(program_path, "r") as f:
                            program_data = json.load(f)
                        program = Program.from_dict(program_data)
                        self.programs[program.id] = program
                    except Exception:
                        pass

    def _save_program(self, program: Program, base_path: Optional[str] = None) -> None:
        save_path = base_path or (self.config.db_path if self.persistence_enabled else None)
        if not save_path:
            return
        programs_dir = os.path.join(save_path, "programs")
        os.makedirs(programs_dir, exist_ok=True)
        program_dict = program.to_dict()
        program_path = os.path.join(programs_dir, f"{program.id}.json")
        with open(program_path, "w") as f:
            json.dump(program_dict, f)


