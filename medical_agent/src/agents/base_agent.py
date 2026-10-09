"""Minimal BaseAgent for medical_text — no JSON parsing, no reflection, no repair."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.llm_client import LLMClient
from src.utils.prompt_loader import PromptLoader
from src.utils.rag_client import RAGClient


class BaseAgent:
    def __init__(
        self,
        task_name: str,
        provider: str,
        model: str,
        temperature: float = 0.3,
        use_rag: bool = True,
    ):
        self.task_name = task_name
        self.provider = provider
        self.model = model
        self.temperature = temperature

        root = Path(__file__).resolve().parents[2]
        self.prompt_loader = PromptLoader(root)
        self.llm_client = LLMClient(
            config_path=root / "config" / "llm_config.yaml",
            provider=provider,
            model=model,
            temperature=temperature,
        )
        self.rag_client = RAGClient(enabled=use_rag) if use_rag else None

    def _load_prompt(self, relative_path: str) -> str:
        try:
            return self.prompt_loader.load_bundle(["shared/output_style_contract.md", relative_path])
        except Exception:
            return self.prompt_loader.load_optional_text(relative_path)

    @staticmethod
    def _strip_think(text: str) -> str:
        return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()

    def _invoke(self, prompt_path: str, payload: Dict[str, Any]) -> str:
        prompt_text = self._load_prompt(prompt_path)
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]
        response = self.llm_client.chat(messages=messages)
        llm_error = self._is_llm_error_content(response.content)
        if llm_error:
            raise RuntimeError(llm_error)
        return self._strip_think(response.content)

    @staticmethod
    def _split_case_input(case_input: dict) -> tuple[dict, str, Dict[str, Any]]:
        if not isinstance(case_input, dict):
            return {}, "", {}
        if "structured_case" in case_input or "case_text" in case_input or "raw_text" in case_input:
            case_flags = dict(case_input.get("case_flags") or {})
            case_flags.setdefault("case_id", str(case_input.get("case_id") or ""))
            return (
                case_input.get("structured_case") or {},
                str(case_input.get("case_text") or case_input.get("raw_text") or ""),
                case_flags,
            )
        return case_input, "", {}

    @staticmethod
    def _is_llm_error_content(content: str) -> Optional[str]:
        try:
            data = json.loads(content)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        message = str(data.get("message", ""))
        if message in {"llm_error", "missing_api_key"}:
            detail = data.get("detail") or data.get("provider") or message
            return f"{message}: {detail}"
        return None

    def _retrieve_context(self, structured_case: dict, raw_text: str = "") -> Dict[str, Any]:
        if self.rag_client is None or not self.rag_client.enabled:
            return {"context": "", "sources": [], "debug": {}}
        self.rag_client.ensure_ready()
        query_plan = RAGClient.build_query_plan(self.task_name, structured_case, raw_text=raw_text)
        if not query_plan:
            return {"context": "", "sources": [], "debug": {}}
        return self.rag_client.retrieve(
            query_plan=query_plan,
            agent_name=self.task_name,
        )

    def execute(self, structured_case: dict) -> str:
        raise NotImplementedError

    def run(self, structured_case: dict, output_dir: Optional[Path] = None, filename: Optional[str] = None, **execute_kwargs) -> Dict[str, Any]:
        started = datetime.now()
        parsed_case, raw_text, case_flags = self._split_case_input(structured_case)
        try:
            content = self.execute(parsed_case, case_text=raw_text, case_flags=case_flags, **execute_kwargs)
            llm_error = self._is_llm_error_content(content)
            if llm_error:
                raise RuntimeError(llm_error)
            status = "completed"
            error = None
        except Exception as exc:
            content = ""
            status = "failed"
            error = str(exc)

        result = {
            "task": self.task_name,
            "status": status,
            "content": content,
            "error": error,
            "started_at": started.isoformat(timespec="seconds"),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "provider": self.provider,
            "model": self.model,
        }

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = filename or self.task_name
            (output_dir / f"{fname}.md").write_text(content or f"<!-- ERROR: {error} -->", encoding="utf-8")
            (output_dir / f"{fname}_meta.json").write_text(
                json.dumps({k: v for k, v in result.items() if k != "content"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return result
