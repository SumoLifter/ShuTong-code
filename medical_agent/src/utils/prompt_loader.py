import json
from pathlib import Path
from typing import Any, Dict


class PromptLoader:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.prompts_root = project_root / "prompts"

    def load_text(self, relative_path: str) -> str:
        target = self.prompts_root / relative_path
        return target.read_text(encoding="utf-8").strip()

    def load_optional_text(self, relative_path: str) -> str:
        target = self.prompts_root / relative_path
        if not target.exists():
            return ""
        return target.read_text(encoding="utf-8").strip()

    def load_bundle(self, relative_paths: list[str]) -> str:
        parts = [self.load_optional_text(path) for path in relative_paths]
        return "\n\n".join(part for part in parts if part).strip()

    def load_json(self, relative_path: str) -> Dict[str, Any]:
        target = self.prompts_root / relative_path
        return json.loads(target.read_text(encoding="utf-8"))
